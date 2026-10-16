import os
import sys
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from ioHandle.MapIO import MapIO
from mapping.config import MappingConfig
from mapping.mapper import IncrementalMapper
from mapping.visualizer import MappingVisualizer


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
    viewer = MappingVisualizer(io.getVisualize())
    completed = False
    try:
        frames = iter(io.load())
        while True:
            load_started = time.perf_counter()
            try:
                frame = next(frames)
            except StopIteration:
                break
            input_load_seconds = time.perf_counter() - load_started
            if not viewer.wait_for_advance(frame.index, frame.name):
                print("[mapping] viewer closed; stopping before the next frame")
                break
            print(f"[mapping] processing frame {frame.index}: {frame.name}")
            result = mapper.process(frame, input_load_seconds=input_load_seconds)
            if viewer.enabled:
                points, colors = mapper.map.to_arrays()
                viewer.update(points, colors, result.status)
        mapper.finalize()
        completed = True
    finally:
        if completed:
            viewer.finish()
        else:
            viewer.close()


if __name__ == "__main__":
    main()
