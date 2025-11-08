import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from ioHandle.FuseIO import FuseIO
from fusion.fusion import BGPatternFuser
from fusion.config import BGPatternFuserConfig

def main():
    io = FuseIO()

    # Config
    cfg = BGPatternFuserConfig(
        hfov_deg=io.cfg["hfov_deg"],
        root_dir=io.getDataRootDir(),
        dst_dir=io.getDstDir(),
        in_camera=io.getInCamera(),
        on_video=io.getOnVideo(),
        do_save=io.getDoSave(),
    )
    fuser = BGPatternFuser(cfg)

    # Main thread work (e.g., check advance flag and update point cloud)
    for dict in io.load():
        # Wait until the flag is set
        fuser.advance.wait()
        print(f"[info] fusing the new frame: {dict['idx']} ...")
        # Generate a random point cloud and update the app
        fuser.fuse(dict["image"],
                   dict["source"],
                   dict["bg_rad"],
                   dict["bg_can"],
                   dict["pose"],
                   dict["name"]
                   )
        print(f"[info] fusion done and scene updated for index {dict['idx']}.")
        # fuser.rescale_and_locate()
        # Reset the advance flag
        fuser.advance.clear()


if __name__ == "__main__":
    main()
