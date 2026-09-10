import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from ioHandle.MapIO import MapIO
from mapping.config import MappingConfig
from mapping.mapper import IncrementalMapper


def main():
    io = MapIO()
    options = io.mapping_options()
    options.setdefault("utm_zone", io.cfg.get("zone"))
    try:
        config = MappingConfig(
            root_dir=io.getDataRootDir(),
            dst_dir=io.getDstDir(),
            do_save=io.getDoSave(),
            **options,
        )
    except TypeError as exc:
        raise SystemExit(f"Invalid key in config.yaml 'mapping' section: {exc}") from exc

    mapper = IncrementalMapper(config)
    for frame in io.load():
        print(f"[mapping] processing frame {frame.index}: {frame.name}")
        mapper.process(frame)
    mapper.finalize()


if __name__ == "__main__":
    main()
