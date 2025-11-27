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
    root = prepare_file_system("data/", "usegeo_1")