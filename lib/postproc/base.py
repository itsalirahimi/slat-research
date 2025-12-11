import argparse
from pathlib import Path

def prepare_file_system(base_path: str, name: str) -> Path:
    root = Path(base_path) / name

    dirs = [
        "diffusion",
        "fusion",
        "projection",
        "eval/gt/in",
        "depth",
        "rgb",
        "pose",
        "raw",
    ]
    files = [
        "config.yaml",
        "data.json",
    ]

    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    for fname in files:
        fpath = root / fname
        fpath.touch(exist_ok=True)

    return root

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create base folders")
    ap.add_argument("--name", required=True, help="database name")
    args = ap.parse_args()
    root = prepare_file_system("data/", args.name)