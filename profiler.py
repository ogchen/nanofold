import argparse
import configparser
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

    return parser.parse_args()


def load_config(filepath):
    config = configparser.ConfigParser()
    with open(filepath) as f:
        config.read_file(f)
    if config.get("General", "device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return config


class ProfilerLogger:
    def __init__(self, prof):
        self.p = prof

    def log_model_summary(self, _):
        pass

    def log_epoch(self, epoch, _):
        self.p.step()

    def log_model(self, model):
        pass

    def log_params(self, params):
        pass


def trace_handler(p):
    output = p.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
    print(output)
    p.export_chrome_trace("/data/trace.json")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.logging.upper()))
    config = load_config(args.config)
    torch.set_default_device(config.get("General", "device"))
    dataset, _ = ChainDataset.construct_datasets(
        args.input,
        1.0,
        config.getint("General", "residue_crop_size"),
        config.getint("General", "num_msa"),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.getint("General", "batch_size")
    )
    next(iter(data_loader))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=10, wait=5, warmup=1, active=5, repeat=1),
        with_stack=True,
        profile_memory=True,
        on_trace_ready=trace_handler,
    ) as prof:
        trainer = Trainer(config, loggers=[ProfilerLogger(prof)], log_every_n_epoch=1)
        trainer.fit(data_loader, {}, max_epoch=40)


if __name__ == "__main__":
    main()
