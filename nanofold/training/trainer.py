import torch

from nanofold.training.model import Nanofold


class Trainer:
    def __init__(
        self,
        params,
        loggers,
        log_every_n_epoch,
        checkpoint_save_freq,
        checkpoint=None,
    ):
        torch.autograd.set_detect_anomaly(
            params["detect_anomaly"],
            check_nan=params["detect_anomaly"],
        )
        self.params = params
        self.loggers = loggers
        self.log_every_n_epoch = log_every_n_epoch
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

    def get_total_loss(self, fape_loss, conf_loss, aux_loss, dist_loss, msa_loss):
        return 0.5 * fape_loss + 0.5 * aux_loss + 0.01 * conf_loss + 0.3 * dist_loss + 2 * msa_loss

    def load_batch(self, batch):
        return {
            k: v.to(self.params["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def training_loop(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            self.params["device"],
            dtype=torch.bfloat16,
            enabled=self.params["use_amp"] and self.params["device"] == "cuda",
        ):
            _, _, _, fape_loss, conf_loss, aux_loss, dist_loss, msa_loss = self.train_model(batch)
            loss = self.get_total_loss(fape_loss, conf_loss, aux_loss, dist_loss, msa_loss)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip_norm"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        with torch.autocast(
            self.params["device"],
            dtype=torch.bfloat16,
            enabled=self.params["use_amp"] and self.params["device"] == "cuda",
        ):
            metrics = {
                "chain_plddt": [],
                "chain_lddt": [],
                "fape_loss": [],
                "conf_loss": [],
                "aux_loss": [],
                "dist_loss": [],
                "msa_loss": [],
                "total_loss": [],
            }

            for _ in range(self.params["num_eval_iters"]):
                _, chain_plddt, chain_lddt, fape_loss, conf_loss, aux_loss, dist_loss, msa_loss = (
                    self.eval_model(self.load_batch(next(iter(loader))))
                )
                total_loss = self.get_total_loss(
                    fape_loss, conf_loss, aux_loss, dist_loss, msa_loss
                )
                metrics["chain_plddt"].append(chain_plddt)
                metrics["chain_lddt"].append(chain_lddt)
                metrics["fape_loss"].append(fape_loss)
                metrics["conf_loss"].append(conf_loss)
                metrics["aux_loss"].append(aux_loss)
                metrics["dist_loss"].append(dist_loss)
                metrics["msa_loss"].append(msa_loss)
                metrics["total_loss"].append(total_loss)

        return {k: torch.stack(v).mean().item() for k, v in metrics.items()}

    def log_epoch(self, epoch, eval_loaders):
        if len(self.loggers) == 0:
            return
        if epoch % self.log_every_n_epoch == 0:
            metrics = {
                f"{k}_{metric_name}": metric
                for k, v in eval_loaders.items()
                for metric_name, metric in self.evaluate(v).items()
            }
            [l.log_epoch(epoch, metrics) for l in self.loggers]
        if epoch % self.checkpoint_save_freq == 0:
            [
                l.log_checkpoint(epoch, self.model, self.optimizer, self.scheduler, self.scaler)
                for l in self.loggers
            ]

    def fit(self, train_loader, eval_loaders, max_epoch):
        for batch in train_loader:
            if self.epoch >= max_epoch:
                break
            self.training_loop(self.load_batch(batch))
            self.log_epoch(self.epoch, eval_loaders)
            self.epoch += 1
        [l.log_model(self.model) for l in self.loggers]
