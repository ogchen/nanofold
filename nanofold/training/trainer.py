import torch

from nanofold.training.model import Nanofold


class Trainer:
    def __init__(self, config, loggers, log_every_n_epoch):
        self.device = config.get("General", "device")
        self.loggers = loggers
        self.log_every_n_epoch = log_every_n_epoch
        params = Nanofold.get_args(config)
        [l.log_params(params) for l in self.loggers]
        self.model = Nanofold(**params)
        self.model = self.model.to(self.device)
        [l.log_model_summary(self.model) for l in self.loggers]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.getfloat("Optimizer", "learning_rate"),
            betas=(
                config.getfloat("Optimizer", "beta1"),
                config.getfloat("Optimizer", "beta2"),
            ),
            eps=config.getfloat("Optimizer", "eps"),
        )

    def load_batch(self, batch):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def training_loop(self, batch):
        _, loss = self.model(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        _, loss = self.model(self.load_batch(next(iter(loader))))
        self.model.train()
        return loss.item()

    def log_epoch(self, epoch, eval_loaders):
        if epoch % self.log_every_n_epoch != 0 or len(self.loggers) == 0:
            return
        metrics = {k: self.evaluate(v) for k, v in eval_loaders.items()}
        [l.log_epoch(epoch, metrics) for l in self.loggers]

    def fit(self, train_loader, eval_loaders, max_epoch):
        epoch = 0
        for batch in train_loader:
            if epoch == max_epoch:
                break
            self.training_loop(self.load_batch(batch))
            self.log_epoch(epoch, eval_loaders)
            epoch += 1
        [l.log_model(self.model) for l in self.loggers]
