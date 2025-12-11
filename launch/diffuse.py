import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from diffusion.diffusion import BGPatternDiffuser, BGPatternDiffuserConfig
from ioHandle.DiffuseIO import DiffuseIO

def main():
    io = DiffuseIO()

    # Config
    cfg = BGPatternDiffuserConfig(
        hfov_deg = io.cfg["hfov_deg"],
        root_dir=io.getDataRootDir(),
        dst_dir=io.getDstDir(),
        do_save=io.getDoSave(),
    )
    diffuser = BGPatternDiffuser(cfg)
    
    for dict in io.load():
        diffuser.diffuse(dict["source"], 
                         dict["image"],
                         dict["name"],
                         dict["idx"]
                         )

if __name__ == '__main__':
    main()