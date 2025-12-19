import os
import shutil

# Input folders
pre_path          = "data/usegeo_2"

rgb_dir           = [f"{pre_path}/projection/test/rgb", "rgb.png", "png"]
m_gep_d_dir       = [f"{pre_path}/fusion/test/gep_mhw1", "m_gep_d.npy", "npy"]
m_gep_p_dir       = [f"{pre_path}/fusion/test/gep_mhw3", "m_gep_p.npy", "npy"]
s_bgcan_d_dir     = [f"{pre_path}/diffusion/test/canonical/bg_mhw1", "s_bgcan_d.npy", "npy"]
s_depthcan_d_dir  = [f"{pre_path}/projection/test/canonical/depthcan_mhw1", "s_depthcan_d.npy", "npy"]
m_agl_dir         = [f"{pre_path}/m_agl", "m_agl.npy", "npy"]
m_gt_dir        = [f"{pre_path}/eval/gt/mhw1_ds", "m_gt.npy", "npy"]
m_reshaped_d_dir  = [f"{pre_path}/fusion/test/fused_mhw1", "m_reshaped_d.npy", "npy"]

out_dir           = f"{pre_path}/sorted"


os.makedirs(out_dir, exist_ok=True)

names = [os.path.splitext(f)[0] for f in os.listdir(rgb_dir[0]) if f.endswith(".png")]

for name in names:
    # Create subfolder for this name
    name_dir = os.path.join(out_dir, name)
    os.makedirs(name_dir, exist_ok=True)

    # Define source and destination paths
    rgb_src = os.path.join(rgb_dir[0], f"{name}.{rgb_dir[2]}")
    m_gep_d_src = os.path.join(m_gep_d_dir[0], f"{name}.{m_gep_d_dir[2]}")
    m_gep_p_src = os.path.join(m_gep_p_dir[0], f"{name}.{m_gep_p_dir[2]}")
    s_bgcan_d_src = os.path.join(s_bgcan_d_dir[0], f"{name}.{s_bgcan_d_dir[2]}")
    s_depthcan_d_src = os.path.join(s_depthcan_d_dir[0], f"{name}.{s_depthcan_d_dir[2]}")
    m_agl_src = os.path.join(m_agl_dir[0], f"{name}.{m_agl_dir[2]}")
    m_gt_src = os.path.join(m_gt_dir[0], f"{name}.{m_gt_dir[2]}")
    m_reshaped_d_src = os.path.join(m_reshaped_d_dir[0], f"{name}.{m_reshaped_d_dir[2]}")

    rgb_dst = os.path.join(name_dir, rgb_dir[1])
    m_gep_d_dst = os.path.join(name_dir, m_gep_d_dir[1])
    m_gep_p_dst = os.path.join(name_dir, m_gep_p_dir[1])
    s_bgcan_d_dst = os.path.join(name_dir, s_bgcan_d_dir[1])
    s_depthcan_d_dst = os.path.join(name_dir, s_depthcan_d_dir[1])
    m_agl_dst = os.path.join(name_dir, m_agl_dir[1])
    m_gt_dst = os.path.join(name_dir, m_gt_dir[1])
    m_reshaped_d_dst = os.path.join(name_dir, m_reshaped_d_dir[1])

    # Copy files if they exist
    if os.path.exists(rgb_src):
        shutil.copy(rgb_src, rgb_dst)
    else:
        print(f"Missing rgb file for {name}")

    if os.path.exists(m_gep_d_src):
        shutil.copy(m_gep_d_src, m_gep_d_dst)
    else:
        print(f"Missing m_gep_d file for {name}")

    if os.path.exists(m_gep_p_src):
        shutil.copy(m_gep_p_src, m_gep_p_dst)
    else:
        print(f"Missing m_gep_p file for {name}")

    if os.path.exists(s_bgcan_d_src):
        shutil.copy(s_bgcan_d_src, s_bgcan_d_dst)
    else:
        print(f"Missing s_bgcan_d file for {name}")
    
    if os.path.exists(s_depthcan_d_src):
        shutil.copy(s_depthcan_d_src, s_depthcan_d_dst)
    else:
        print(f"Missing s_depthcan_d file for {name}")
    
    if os.path.exists(m_agl_src):
        shutil.copy(m_agl_src, m_agl_dst)
    else:
        print(f"Missing m_agl file for {name}")
    
    if os.path.exists(m_gt_src):
        shutil.copy(m_gt_src, m_gt_dst)
    else:
        print(f"Missing m_gt file for {name}")
    
    if os.path.exists(m_reshaped_d_src):
        shutil.copy(m_reshaped_d_src, m_reshaped_d_dst)
    else:
        print(f"Missing m_reshaped_d file for {name}")
    
    print(f"Saved files for {name} in {name_dir}")
