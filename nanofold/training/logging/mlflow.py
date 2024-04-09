from pathlib import Path
import mlflow
import tempfile
import torchinfo

from nanofold.training.logging import Logger


class MLFlowLogger(Logger):
    def __init__(self, uri, pip_requirements):
        super().__init__()
        self.pip_requirements = pip_requirements
        mlflow.set_tracking_uri(uri=uri)
        mlflow.start_run()

    def __del__(self):
        mlflow.end_run()

    def log_model_summary(self, model):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_summary = Path(tmp_dir) / "model_summary.txt"
            model_summary.write_text(str(torchinfo.summary(model, verbose=0)))
            mlflow.log_artifact(model_summary)

    def log_epoch(self, epoch, metrics):
        for k, metric in metrics.items():
            mlflow.log_metric(k, metric, step=epoch)

    def log_model(self, model):
        mlflow.pytorch.log_model(model, "model", pip_requirements=self.pip_requirements)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_config(self, config_dict):
        mlflow.log_dict(config_dict, "config.json")
