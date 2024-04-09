import mlflow
import torch
from pathlib import Path
from tempfile import TemporaryDirectory


class CheckpointLoader:
    def __init__(self, uri, run_id):
        mlflow.set_tracking_uri(uri=uri)
        self.run = mlflow.get_run(run_id)
        self.checkpoints = sorted(
            [
                (int(c.path.split("_")[-1]), c)
                for c in mlflow.artifacts.list_artifacts(
                    run_id=self.run.info.run_id, artifact_path="checkpoints"
                )
            ],
            key=lambda c: c[0],
        )

    def get_params(self):
        return mlflow.artifacts.load_dict(self.run.info.artifact_uri + "/config.json")

    def get_checkpoint(self, epoch=None):
        epoch = self.checkpoints[-1][0] if epoch is None else epoch
        checkpoint = dict(self.checkpoints)[epoch]

        with TemporaryDirectory() as tmp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=self.run.info.run_id, artifact_path=checkpoint.path, dst_path=str(tmp_dir)
            )
            return torch.load(Path(tmp_dir) / checkpoint.path)
