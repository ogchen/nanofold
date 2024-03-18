import pyarrow as pa


def load_table(path):
    if path.exists():
        with pa.OSFile(str(path), mode="rb") as f:
            return pa.ipc.open_file(f).read_all()
    return None


def write_table(filepath, table, schema):
    with pa.OSFile(str(filepath), mode="w") as f:
        with pa.ipc.new_file(f, schema) as writer:
            writer.write_table(table)
