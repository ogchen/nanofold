import gzip
import logging
import os
import subprocess
from pathlib import Path


class HHSearchRunner:
    def __init__(self, hhsearch_bin, pdb70, cache_dir):
        self.hhsearch_bin = hhsearch_bin
        self.pdb70 = pdb70
        self.cache_dir = cache_dir

    def build_cmd(self, a2m_file, output):
        return [
            self.hhsearch_bin,
            "-i",
            a2m_file,
            "-o",
            output,
            "-d",
            self.pdb70,
            "-v",
            "0",
            "-cpu",
            "1",
        ]

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
        output = self.cache_dir / f"{id}.hhr"
        cached_result = self.cached_result(output)
        if cached_result is not None:
            return cached_result

        cmd = self.build_cmd(a2m_file, output)
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(e.stderr)
            raise e

        return self.cached_result(output)
