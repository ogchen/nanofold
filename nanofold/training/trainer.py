import torch

from nanofold.training.model import Nanofold


class Trainer:
    def __init__(self, config, loggers, log_every_n_epoch):
        self.loggers = loggers
        self.log_every_n_epoch = log_every_n_epoch
        params = Nanofold.get_args(config)
        [l.log_params(params) for l in self.loggers]
        self.model = Nanofold(**params)
        self.model = self.model.to(config.get("General", "device"))
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

    def get_total_loss(self, fape_loss, conf_loss, aux_loss, dist_loss):
        return 0.5 * fape_loss + 0.5 * aux_loss + 0.01 * conf_loss + 0.3 * dist_loss

    def training_loop(self, batch):
        _, _, _, fape_loss, conf_loss, aux_loss, dist_loss = self.model(batch)
        self.optimizer.zero_grad()
        self.get_total_loss(fape_loss, conf_loss, aux_loss, dist_loss).backward()
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        _, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss, dist_loss = self.model(
            next(iter(loader))
        )
        self.model.train()
        return {
            "chain_plddt": chain_plddt.mean().item(),
            "chain_lddt": chain_lddt.mean().item(),
            "fape_loss": fape_loss.item(),
            "conf_loss": conf_loss.item(),
            "aux_loss": aux_loss.item(),
            "dist_loss": dist_loss.item(),
            "total_loss": self.get_total_loss(fape_loss, conf_loss, aux_loss, dist_loss).item(),
        }

    def log_epoch(self, epoch, eval_loaders):
        if epoch % self.log_every_n_epoch != 0 or len(self.loggers) == 0:
            return
        metrics = {
            f"{k}_{metric_name}": metric
            for k, v in eval_loaders.items()
            for metric_name, metric in self.evaluate(v).items()
        }
        [l.log_epoch(epoch, metrics) for l in self.loggers]

    def fit(self, train_loader, eval_loaders, max_epoch):
        epoch = 0
        for batch in train_loader:
            if epoch == max_epoch:
                break
            self.training_loop(batch)
            self.log_epoch(epoch, eval_loaders)
            epoch += 1
        [l.log_model(self.model) for l in self.loggers]
