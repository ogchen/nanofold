import logging
import torchinfo


class Logger:
    def log_model_summary(self, model):
        logging.info(str(torchinfo.summary(model, verbose=0)))

    def log_epoch(self, epoch, metrics):
        metrics_str = ", ".join([f"{k}: {metric}" for k, metric in metrics.items()])
        logging.info(f"Epoch {epoch}: {metrics_str}")

    def log_model(self, model):
        pass

    def log_params(self, params):
        pass
