
import os
from ioHandle.IOHandler import IOHandler, load_pcd

class DiffuseIO(IOHandler):
    def __init__(self):
        super().__init__()
        self.source_dir = self.getSrcDir()
        self.dataset_dir = self.source_dir.expanduser().resolve().parent.parent.parent
        parent = os.path.basename(self.source_dir)
        test_name = os.path.basename(os.path.dirname(self.source_dir))
        self.dst = test_name + "/" + parent
        self.init_dataset(self.dataset_dir)
    
    def load_row(self, row):

        idx = row['index']
        name = row["name"]
        img_dir = os.path.join(self.source_dir, "..", "rgb")
        color_img = self.get_color_image(img_dir , name, self.cfg["rgb_extension"])
        projected = load_pcd(os.path.join(self.source_dir, f"{name}.pcd"))
         
        dic = {
            "idx": idx,
            "name": name,
            "source": projected, 
            "image": color_img,
        }
        return dic
