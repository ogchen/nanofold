import logging
import torchinfo


class Logger:
    def __init__(self, log_every_n_epoch):
        self.log_every_n_epoch = log_every_n_epoch

    def log_model_summary(self, model):
        logging.info(str(torchinfo.summary(model, verbose=0)))

    def log_epoch(self, epoch, train_metrics, test_metrics):
        if epoch % self.log_every_n_epoch != 0:
            return
        metric_ordering = [
            "total_loss",
            "dist_loss",
            "dist_coords_loss",
            "fape_loss",
            "aux_loss",
            "msa_loss",
            "conf_loss",
            "chain_plddt",
            "chain_lddt",
        ]
        metrics = [
            f"{m} | train: {train_metrics.get(m):.4f}, test: {test_metrics.get(m):.4f}"
            for m in metric_ordering
        ]
        metrics_str = "\n".join([f"  {m}" for m in metrics])
        logging.info(f"Epoch {epoch}:\n{metrics_str}")

    def log_model(self, model):
        pass

    def log_params(self, params):
        pass

    def log_config(self, config_dict):
        pass

    def log_checkpoint(self, epoch, model, optimizer, scheduler, scaler):
        pass
