import argparse
import json
import logging
import torch
from pathlib import Path

from nanofold.training.chain_dataset import ChainDataset
from nanofold.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file for profiling")
    parser.add_argument(
        "-i", "--input", help="Input chain training data in Arrow IPC file format", type=Path
    )
    parser.add_argument("-l", "--logging", help="Logging level", default="INFO")
    parser.add_argument(
        "--mode", help="Mode of operation", choices=["time", "memory"], action="append"
    )

    return parser.parse_args()


def load_config(filepath):
    with open(filepath) as f:
        params = json.load(f)
    if params["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return params


class ProfiledTrainer(Trainer):
    def __init__(self, prof, params, *args, **kwargs):
        self.prof = prof
        super().__init__(params, *args, **kwargs)

    def training_loop(self, *args, **kwargs):
        super().training_loop(*args, **kwargs)
        self.prof.step()


def trace_handler(p):
    p.export_chrome_trace("/data/trace.json")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    params = load_config(args.config)
    params["disable_scaler"] = True
    dataset, _ = ChainDataset.construct_datasets(
        args.input,
        1.0,
        params["residue_crop_size"],
        params["num_msa_clusters"],
        params["num_extra_msa"],
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    if "time" in args.mode:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=1, repeat=1),
            with_stack=True,
            profile_memory=True,
            on_trace_ready=trace_handler,
        ) as prof:
            trainer = ProfiledTrainer(prof, params, loggers=[], checkpoint_save_freq=1)
            trainer.fit(data_loader, None, max_epoch=6)
    if "memory" in args.mode:
        for _ in range(50):
            batch = next(iter(data_loader))
            if (
                len(batch["extra_msa_feat"]) == params["num_extra_msa"]
                and len(batch["target_feat"]) == params["residue_crop_size"]
            ):
                break
        torch.cuda.memory._record_memory_history(max_entries=100000)
        trainer = Trainer(params, loggers=[], checkpoint_save_freq=1)
        trainer.fit([batch], None, max_epoch=1)
        torch.cuda.memory._dump_snapshot("/data/snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
