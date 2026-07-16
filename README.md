## Major Job

```bash
python3 launch/project.py --src data/bvc/depth/depth_pro/ --dst test --save --start 50
python3 launch/diffuse.py --src data/bvc/projection/test/canonical/ --save --start 50
```

### ORTHOLOC
```bash
# projection


python3 lib/postproc/ortholoc1.py --raw data/ort/raw/ --out data/ort/
python3 lib/postproc/ortholoc2.py --src data/ort/eval/gt/npz_ds/


```
### USEGEO
```bash
# -------------Step 1 - Post Process
# first run: to create base folders in data/usegeo_1
python3 lib/postproc/base.py --name usegeo_1

# Then move Depth_resized/depth_maps and Depth_resized/undistorted_images to /raw 
# remove "_depth_res" from .tiff files and remove "_res" from .jpg files
# move Image_orientations_dataset[NUMBER].xyz to raw/Image_orientations_dataset.xyz

# This launch file create what we need from raw data of usegeo
python3 lib/postproc/usegeo_1.py --raw data/usegeo_1/raw/ --out data/usegeo_1/

# Project LiDAR (write eval/gt/pcd_ds)
python3 lib/postproc/usegeo_2.py --src data/usegeo_1/eval/gt/npz_ds/


# -------------Step 2 - gt diffusion (write diffusion/gt/pcd_ds)
python3 launch/diffuse.py --src data/usegeo_1/eval/gt/pcd_ds/ --save

# -------------Step 3 - Estimate AGL (overwrite agl on data.json)
python3 lib/postproc/estimateAGL.py --src data/usegeo_1/diffusion/gt/pcd_ds/background/

# -------------Step 4 - Project depth
python3 launch/project.py --src data/usegeo_1/depth/depth_pro/ --dst test --save

# -------------Step 5 - run diffusion
python3 launch/diffuse.py --src data/usegeo_1/projection/test/radial/ --save
python3 launch/diffuse.py --src data/usegeo_1/projection/test/canonical/ --save


# -------------Step 6 - run fusion
python3 launch/fuse.py --src data/usegeo_1/projection/test/radial/ --save

# -------------Step 7 - evaluation


```

### Run Diffusion for different MDEs
```bash
python3 lib/postproc/base.py --name dtm
python3 launch/project.py --src data/dtm/depth/depthpro/ --dst depthpro --save DONE
python3 launch/project.py --src data/dtm/depth/zoedepth/ --dst zoedepth --save DONE
python3 launch/project.py --src data/dtm/depth/marigold/ --dst marigold --save DONE
python3 launch/project.py --src data/dtm/depth/da2/ --dst da2 --save DONE
python3 launch/project.py --src data/dtm/depth/da3/ --dst da3 --save DONE

python3 launch/diffuse.py --src data/dtm/projection/depthpro/canonical/ --save DONE
python3 launch/diffuse.py --src data/dtm/projection/zoedepth/canonical/ --save DONE
python3 launch/diffuse.py --src data/dtm/projection/marigold/canonical/ --save
python3 launch/diffuse.py --src data/dtm/projection/da2/canonical/ --save
python3 launch/diffuse.py --src data/dtm/projection/da3/canonical/ --save
```

Flags with help
```bash
--index INDEX      Row index to use from dataset (matches 'index' column)
--src DATASET      Path to the source directory (e.g., ../../data/e)
--start START      Row index to start from
--save             To save the output in a corresponding dir name
--on-video         Perform position translation on each projected point cloud
--in-camera        Perform projection in all of the subroutines, in camera frame. ATTENTION: YOU MUST SET THIS FLAG BOTH IN FUSION AND DIFFUSION OTHERWISE CORRUPTED OUTPUTS!
```

**Scaling Fixed**
Check the scaling and it is now 'MEAN_Z', which finds the scale factor such that the distance of flattened background equals the pose altitude. Must be much better than 'MIN_Z' (the old method) which tried to equalize the minimun z of raw projected depth to pose altitude - prone to great errors when there is tiny a error in ground perception by MDE.

**Shape Enhanced**
The current method for reshaping is: project depth radially, diffuse background spline, reshape depth (currently NDFDrop method is tested) to flatten the background. The shapes ar better than raw projected MDE output (which basically needs pyramid projection). 

### Research Trials
**Question:** Wasn't it cleaner if we de-canonicalized the depth pyramid with a simpler geometrical transform? Was its quality equal to what we have now? If ours would be better, then our current math is something worth it! ...
**Update/Answer:** I added the 'depyramidization' method in fusion methods and compared it with our currently active NDFDrop method. Turns out that our method is better. Indeed, to depyramidize the raw depth's pyramid-projection, we need backgroud info. Alternatively, the method is highly dependent to the z-val of flat background used as rotation anchor for any point (See function `fusion.helper.depyramidize_pointCloud` --> i'm talking about the var: `plane_z` in that function). Even determining such a plane as a planar mean of diffused background does not guarantee that the resulting reshaping method (so far, it must have been much more complex-in-implementation than my NDFDrop method!) will be consistent in reshaping the point cloud geometry in different areas (based on the fact that the current point to be depyramidized is below, or above the backgroud-mean-z plane), and probably ruins the local shapes. 
**Conclusion:** So yes! This proves that our flat-ground-fusion method is what it must be


**Question:** The current fine-tunning in diffusion procedure takes too much time. Isn't it better to satisfy with corse-tunning and fit a higher-degree spline surface to a filtered set of points after coarse-tunning, as fine-tunning without GD??
**Update/Answer:** In the previous commint, I implemented the above idea. The `research/diffusion/spline2d/fine_tune_fit.py` proves that surface fitting on filtered data is much weaker than our current iterative tunning method. If fine-tunning takes time, we can simply make the fine-tune spline grid weaker, which is not suggested! Fine-tunning tries to fit the ground patter more, in a tiny optimization radius which does not allow the ctrl points to see the non-ground points of point cloud (i.e. `max_dz` is set tiny in `scorer.reset` in fine-tunning phase of `diffusion.diffusion. ... .diffuse`)
**Conclusion:** So this proves that we are good to iteratively estimate the background pattern in our dataset generation scheme.


## Evaluation

Currently, use something like:
```bash
# in the root of a dataset, after running diffusion and fusion
python3 tools/vis/compare_pcds.py fusion/something.pcd red rawdepth/something.pcd
```

**TODO:**
Provide an automatic evaluation through ground-truth data and our output 3d models. Separately check the following metrics:
- Shape error (Independent of scale and pose)
- Scale error (Independent of shape and pose)
- Pose error (Independent of shape and scale)
- Depth error (calculated in 2.5 depth image space)
- Point cloud error (point-by-point distance)

### Final Products
**Finalize the method**
1. Mean_z scaling + NDFDrop reshape (current state)
2. (above) + Does 'unfold' make the results better?
--> The outcome of 1,2,3 is our product-1

**Background RLS**

4. Evaluate the product-1 using a background model fitted overally on the vision stream (in an online scheme: correcting the ground model through sequential vision stream) 

--> product-2 (removes product-1)

**Elevation Fusion**

5. Instead of flat-ground-fusion, perform elevation data fusion with elevation diffused from GT data OR nasa elevation data

--> product-3 (presentable along with product-2)