import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from projection.mapper3D import Mapper3D, Mapper3DConfig
from ioHandle.ProjcetIO import ProjcetIO

def main():
    io = ProjcetIO()

    # Config
    cfg = Mapper3DConfig(color_mode='image', # Options: 'image', 'proximity', 'constant', 'none'
                         hfov_deg=io.cfg["hfov_deg"],
                         root_dir=io.getDataRootDir(),
                         dst_dir=io.getDstDir(),
                         do_save=io.getDoSave(),
                         on_video=io.getOnVideo())
    mapper = Mapper3D(cfg)

    # Main thread work (e.g., check advance flag and update point cloud)
    for dict in io.load():
        # Wait until the flag is set
        mapper.advance.wait()
        print(f"[info] projecting the new frame: {dict['idx']} ...")
        # Generate a random point cloud and update the app
        mapper.project(dict["source"],
                       dict["pose"],
                       dict["image"],
                       dict["name"]
                       )
        
        print(f"[info] projection done and scene updated for index {dict['idx']}.")
        # Reset the advance flag
        mapper.advance.clear()

if __name__ == "__main__":
    main()
