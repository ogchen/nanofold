import torch

from nanofold.training.model import Nanofold


class Trainer:
    def __init__(
        self,
        params,
        loggers,
        checkpoint_save_freq,
        checkpoint=None,
    ):
        torch.autograd.set_detect_anomaly(
            params["detect_anomaly"],
            check_nan=params["detect_anomaly"],
        )
        self.params = params
        self.loggers = loggers
        self.checkpoint_save_freq = checkpoint_save_freq
        self.setup_model(checkpoint)
        [l.log_params(params) for l in self.loggers]
        [l.log_config(params) for l in self.loggers]
        [l.log_model_summary(self.model) for l in self.loggers]

    def setup_model(self, checkpoint):
        self.model = Nanofold(**Nanofold.get_args(self.params))
        self.model = self.model.to(self.params["device"])
        compile_model = lambda m: torch.compile(
            m,
            disable=not self.params["compile_model"],
            dynamic=False,
            mode=self.params.get("compilation_mode", "default"),
        )
        self.train_model = compile_model(self.model)
        self.eval_model = compile_model(self.model)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            betas=(self.params["beta1"], self.params["beta2"]),
            eps=self.params["optimizer_eps"],
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.params["lr_start_factor"],
            total_iters=self.params["lr_warmup"],
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.params["use_amp"] and self.params["device"] == "cuda",
        )
        self.epoch = 0
        if checkpoint is not None:
            self.epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def load_batch(self, batch):
        return {
            k: v.to(self.params["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def training_loop(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            self.params["device"],
            enabled=self.params["use_amp"] and self.params["device"] == "cuda",
        ):
            out = self.train_model(batch)
        self.scaler.scale(out["total_loss"]).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip_norm"])
        self.scaler.step(self.optimizer)
        current_scale = self.scaler.get_scale()
        self.scaler.update()
        if current_scale <= self.scaler.get_scale():
            self.scheduler.step()
        return {k: v.item() for k, v in out.items() if k != "coords"}

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()
        with torch.autocast(
            self.params["device"],
            enabled=self.params["use_amp"] and self.params["device"] == "cuda",
        ):
            out = self.eval_model(batch)
        self.model.train()
        return {k: v.item() for k, v in out.items() if k != "coords"}

    def log_epoch(self, train_metrics, test_metrics):
        [l.log_epoch(self.epoch, train_metrics, test_metrics) for l in self.loggers]

    def save_checkpoint(self):
        if self.epoch % self.checkpoint_save_freq == 0:
            [
                l.log_checkpoint(
                    self.epoch, self.model, self.optimizer, self.scheduler, self.scaler
                )
                for l in self.loggers
            ]

    def fit(self, train_loader, test_loader, max_epoch):
        while True:
            if self.epoch >= max_epoch:
                break
            train_metrics = self.training_loop(self.load_batch(next(iter(train_loader))))
            if (
                any([self.epoch % l.log_every_n_epoch == 0 for l in self.loggers])
                and test_loader is not None
            ):
                test_metrics = self.evaluate(self.load_batch(next(iter(test_loader))))
                [l.log_epoch(self.epoch, train_metrics, test_metrics) for l in self.loggers]
            self.epoch += 1
            self.save_checkpoint()
        [l.log_model(self.model) for l in self.loggers]
