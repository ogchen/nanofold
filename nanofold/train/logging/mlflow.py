import mlflow
import tempfile
import torch
import torchinfo
from pathlib import Path

from nanofold.train.logging import Logger


class MLFlowLogger(Logger):
    def __init__(self, uri, pip_requirements, log_every_n_epoch, run_id=None):
        super().__init__(log_every_n_epoch)
        self.pip_requirements = pip_requirements
        mlflow.set_tracking_uri(uri=uri)
        mlflow.start_run(run_id=run_id)

    def __del__(self):
        mlflow.end_run()

    def log_model_summary(self, model):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_summary = Path(tmp_dir) / "model_summary.txt"
            model_summary.write_text(str(torchinfo.summary(model, verbose=0)))
            mlflow.log_artifact(model_summary)

    def log_epoch(self, epoch, train_metrics, test_metrics):
        if epoch % self.log_every_n_epoch != 0:
            return
        metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {
            f"test_{k}": v for k, v in test_metrics.items()
        }
        mlflow.log_metrics(metrics, step=epoch)

    def log_model(self, model):
        mlflow.pytorch.log_model(model, "model", pip_requirements=self.pip_requirements)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_config(self, config_dict):
        mlflow.log_dict(config_dict, "config.json")

    def log_checkpoint(self, epoch, model, optimizer, scheduler, scaler):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / f"checkpoint_{epoch}"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                filepath,
            )
            mlflow.log_artifact(filepath, artifact_path="checkpoints")
