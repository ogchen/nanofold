import torch

from nanofold.training.model import Nanofold


class Trainer:
    def __init__(self, config, loggers, log_every_n_epoch):
        detect_anomaly = config.getboolean("General", "detect_anomaly")
        torch.autograd.set_detect_anomaly(detect_anomaly, check_nan=detect_anomaly)
        self.device = config.get("General", "device")
        self.use_amp = config.getboolean("General", "use_amp") and self.device == "cuda"
        self.loggers = loggers
        self.log_every_n_epoch = log_every_n_epoch
        self.setup_model(config)
        self.clip_norm = config.getfloat("Nanofold", "clip_norm")

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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def setup_model(self, config):
        params = Nanofold.get_args(config)
        [l.log_params(params) for l in self.loggers]
        self.model = Nanofold(**params)
        self.model = self.model.to(self.device)
        compile_model = lambda m: torch.compile(
            m,
            disable=not config.getboolean("General", "compile_model"),
            dynamic=False,
            mode=config.get("General", "compilation_mode", fallback="default"),
        )
        self.train_model = compile_model(self.model)
        self.eval_model = compile_model(self.model)

    def get_total_loss(self, fape_loss, conf_loss, aux_loss, dist_loss):
        return 0.5 * fape_loss + 0.5 * aux_loss + 0.01 * conf_loss + 0.3 * dist_loss

    def load_batch(self, batch):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def training_loop(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.use_amp):
            _, _, _, fape_loss, conf_loss, aux_loss, dist_loss = self.train_model(batch)
            loss = self.get_total_loss(fape_loss, conf_loss, aux_loss, dist_loss)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.use_amp):
            _, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss, dist_loss = self.eval_model(
                self.load_batch(next(iter(loader)))
            )
            loss = self.get_total_loss(fape_loss, conf_loss, aux_loss, dist_loss)
        self.model.train()
        return {
            "chain_plddt": chain_plddt.mean().item(),
            "chain_lddt": chain_lddt.mean().item(),
            "fape_loss": fape_loss.item(),
            "conf_loss": conf_loss.item(),
            "aux_loss": aux_loss.item(),
            "dist_loss": dist_loss.item(),
            "total_loss": loss.item(),
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
            self.training_loop(self.load_batch(batch))
            self.log_epoch(epoch, eval_loaders)
            epoch += 1
        [l.log_model(self.model) for l in self.loggers]
