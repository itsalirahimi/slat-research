import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from evaluation.Evaluator3D import Evaluator3D
from evaluation.config import EvalConfig
from ioHandle.EvalIO import EvalIO

def main():
    io = EvalIO()

    cfg = EvalConfig(
                    root_dir=io.getDataRootDir(),
                    dst_dir=io.getDstDir(),
                    do_save=io.getDoSave(),
                    on_video=io.getOnVideo())

    e = Evaluator3D(cfg)

    

    for dict in io.load():
        # Wait until the flag is set
        e.advance.wait()
        print(f"[info] projecting the new frame: {dict['idx']} ...")
        e.eval(
            ref=dict["groundtruth"],
            test=dict["fused"],
            filename=dict["name"],
            depth_eval=False
        )
        
        e.dry_run()
        print(f"[info] projection done and scene updated for index {dict['idx']}.")
        # Reset the advance flag
        e.advance.clear()



if __name__ == "__main__":
    main()
