import logging
import torchinfo


class Logger:
    def log_model_summary(self, model):
        logging.info(str(torchinfo.summary(model, verbose=0)))

    def log_epoch(self, epoch, metrics):
        metric_keys = [k.split("_", maxsplit=1) for k in metrics.keys()]
        categories = list(reversed(sorted(set(k[0] for k in metric_keys))))
        metric_names = set(k[1] for k in metric_keys)
        loss_names = [m for m in metric_names if "loss" in m]
        metric_names = sorted(loss_names) + sorted(metric_names - set(loss_names))

        ordered_metrics = []
        for m in metric_names:
            keys = [f"{c}_{m}" for c in categories]
            ordered_metrics.append(
                f"{m} | " + ", ".join([f"{c}: {metrics[c + '_' + m]:.4f}" for c in categories])
            )

        metrics_str = "\n".join([f"  {m}" for m in ordered_metrics])
        logging.info(f"Epoch {epoch}:\n{metrics_str}")

    def log_model(self, model):
        pass

    def log_params(self, params):
        pass
