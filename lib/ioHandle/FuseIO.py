import os
from ioHandle.IOHandler import IOHandler, load_pcd

class FuseIO(IOHandler):
    def __init__(self):
        super().__init__()
        self.source_dir = self.getSrcDir()
        self.dataset_dir = self.source_dir.expanduser().resolve().parent.parent.parent
        self.dst = os.path.basename(os.path.dirname(self.source_dir))
        self.init_dataset(self.dataset_dir)
    
    def load_row(self, row):

        idx = row['index']
        name = row["name"]
        pose = self.get_pose(row)
        
        img_dir = os.path.join(self.dataset_dir, "projection", self.dst, "rgb")
        color_img = self.get_color_image(img_dir , name, self.cfg["rgb_extension"])

        bg_can_dir = os.path.join(self.dataset_dir, "diffusion", self.dst, "canonical", "background")
        bg_can = load_pcd(os.path.join(bg_can_dir, f"{name}.pcd"))

        bg_rad_dir = os.path.join(self.dataset_dir, "diffusion", self.dst, "radial", "background")
        bg_rad = load_pcd(os.path.join(bg_rad_dir, f"{name}.pcd"))

        prj_can_dir = os.path.join(self.dataset_dir, "projection", self.dst, "canonical")
        prj_can = load_pcd(os.path.join(prj_can_dir, f"{name}.pcd"))

        prj_rad_dir = os.path.join(self.dataset_dir, "projection", self.dst, "radial")
        prj_rad = load_pcd(os.path.join(prj_rad_dir, f"{name}.pcd"))

        dic = {
            "idx": idx,
            "name": name,
            "pose": pose, 
            "prj_can": prj_can, 
            "prj_rad": prj_rad, 
            "bg_can": bg_can, 
            "bg_rad": bg_rad, 
            "image": color_img,
        }
        return dic