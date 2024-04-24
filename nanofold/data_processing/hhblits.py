import gzip
import os
import subprocess
from pathlib import Path


class HHblitsRunner:
    def __init__(
        self, bin, db, cache_dir, num_iterations, output_format="hhr", num_cpu=1, **kwargs
    ):
        self.bin = bin
        self.db = db
        self.cache_dir = cache_dir
        self.num_iterations = num_iterations
        self.num_cpu = num_cpu
        self.output_format = output_format
        self.kwargs = kwargs

    def build_cmd(self, a2m_file, output):
        cmd = [
            self.bin,
            "-i",
            a2m_file,
            "-d",
            self.db,
            "-v",
            "0",
            "-cpu",
            self.num_cpu,
            "-n",
            self.num_iterations,
        ]
        if self.output_format == "hhr":
            cmd.extend(["-o", output])
        elif self.output_format == "a3m":
            cmd.extend(["-oa3m", output])
        for k, v in self.kwargs.items():
            cmd.extend([f"-{k}", v])
        return [str(c) for c in cmd]

    def cached_result(self, output):
        zip_output = Path(f"{output}.gz")
        content = None

        if zip_output.exists():
            with gzip.open(zip_output, "rb") as gz_f:
                content = gz_f.read().decode()
        if output.exists() and os.path.getsize(output) > 0:
            with open(output) as f:
                content = f.read()
                with gzip.open(zip_output, "wb") as gz_f:
                    gz_f.write(content.encode())
            os.remove(output)
        return content

    def run(self, a2m_file, id):
        output = self.cache_dir / f"{id}.{self.output_format}"
        cached_result = self.cached_result(output)
        if cached_result is not None:
            return cached_result

        cmd = self.build_cmd(a2m_file, output)
        subprocess.run(cmd, check=True, text=True)

        return self.cached_result(output)
