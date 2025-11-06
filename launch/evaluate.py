import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from projection.mapper3D import Evaluator3D
from ioHandle.IOHandler import IOHandler

def main():
    io = IOHandler(True)

    e = Evaluator3D()

    for dict in io.load():
        # Wait until the flag is set
        e.advance.wait()
        print(f"[info] projecting the new frame: {dict['idx']} ...")
        # Generate a random point cloud and update the app
        e.eval()
        print(f"[info] projection done and scene updated for index {dict['idx']}.")
        # Reset the advance flag
        e.advance.clear()



if __name__ == "__main__":
    main()
