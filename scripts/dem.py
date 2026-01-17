import os
import argparse
import math
import requests
import numpy as np

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import xy

from pystac_client import Client
import planetary_computer as pc
import open3d as o3d


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "cop-dem-glo-30"  # Copernicus DEM GLO-30 on Planetary Computer :contentReference[oaicite:1]{index=1}
ASSET_KEY = "data"            # Collection declares item_assets "data" :contentReference[oaicite:2]{index=2}


def bbox_from_center(lon: float, lat: float, radius_km: float):
    """Approx bbox around center (good for small areas). Returns (min_lon, min_lat, max_lon, max_lat)."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def download_file(url: str, out_path: str, chunk_size: int = 1024 * 1024):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path


def search_dem_tiles(bbox_4326):
    """Search STAC for DEM tiles intersecting bbox."""
    catalog = Client.open(STAC_URL)  # Planetary Computer STAC API :contentReference[oaicite:3]{index=3}
    search = catalog.search(collections=[COLLECTION], bbox=list(bbox_4326))
    items = list(search.items())
    return items


def mosaic_and_clip(items, bbox_4326, out_dem_tif, cache_dir="data/_tiles"):
    """
    Download tiles, mosaic, clip to bbox, write GeoTIFF.
    bbox_4326: (minlon, minlat, maxlon, maxlat) in EPSG:4326.
    """
    if not items:
        raise RuntimeError("No DEM tiles found for that area (try a larger radius/bbox).")

    # Sign items for read access :contentReference[oaicite:4]{index=4}
    signed_items = [pc.sign(it) for it in items]

    local_paths = []
    for it in signed_items:
        if ASSET_KEY not in it.assets:
            raise RuntimeError(f"Item {it.id} missing asset '{ASSET_KEY}'. Available: {list(it.assets.keys())}")
        url = it.assets[ASSET_KEY].href
        local_path = os.path.join(cache_dir, f"{it.id}.tif")
        local_paths.append(download_file(url, local_path))

    # Mosaic
    srcs = [rasterio.open(p) for p in local_paths]
    try:
        mosaic, out_transform = merge(srcs)
        profile = srcs[0].profile.copy()
        crs = srcs[0].crs
        nodata = profile.get("nodata", None)
    finally:
        for s in srcs:
            s.close()

    # Write mosaic to temp
    tmp_mosaic = out_dem_tif.replace(".tif", "_mosaic_tmp.tif")
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=out_transform,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    os.makedirs(os.path.dirname(out_dem_tif) or ".", exist_ok=True)
    with rasterio.open(tmp_mosaic, "w", **profile) as dst:
        dst.write(mosaic)

    # Clip to bbox (transform bbox into dataset CRS if needed)
    with rasterio.open(tmp_mosaic) as src:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            b = transform_bounds("EPSG:4326", src.crs, *bbox_4326, densify_pts=21)
            minx, miny, maxx, maxy = b
        else:
            minx, miny, maxx, maxy = bbox_4326

        geom = [{
            "type": "Polygon",
            "coordinates": [[
                (minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)
            ]]
        }]

        clipped, clipped_transform = mask(src, geom, crop=True, nodata=src.nodata)

        clipped_profile = src.profile.copy()
        clipped_profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

    with rasterio.open(out_dem_tif, "w", **clipped_profile) as dst:
        dst.write(clipped)

    # Cleanup temp mosaic
    try:
        os.remove(tmp_mosaic)
    except OSError:
        pass

    return out_dem_tif


def reproject_dem_to_utm(dem_path: str):
    """Reproject DEM to UTM (meters) for nicer point spacing."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform
        b = src.bounds

        # Find center lon/lat to pick UTM zone robustly
        if src_crs and src_crs.to_string() != "EPSG:4326":
            lonlat = transform_bounds(src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
            center_lon = (lonlat[0] + lonlat[2]) / 2.0
            center_lat = (lonlat[1] + lonlat[3]) / 2.0
        else:
            center_lon = (b.left + b.right) / 2.0
            center_lat = (b.bottom + b.top) / 2.0

        dst_crs = f"EPSG:{utm_epsg_from_lonlat(center_lon, center_lat)}"
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        dem_utm = np.empty((dst_h, dst_w), dtype=np.float32)

        reproject(
            source=dem,
            destination=dem_utm,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    return dem_utm, dst_transform, dst_crs, nodata


def dem_to_points(dem: np.ndarray, transform, nodata=None, stride: int = 3):
    """DEM grid -> XYZ points."""
    h, w = dem.shape
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    cc, rr = np.meshgrid(cols, rows)

    z = dem[rr, cc].astype(np.float32)

    m = np.isfinite(z)
    if nodata is not None:
        m &= (z != nodata)

    rr = rr[m]
    cc = cc[m]
    z = z[m]

    xs, ys = xy(transform, rr.tolist(), cc.tolist(), offset="center")
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    return np.column_stack([x, y, z]).astype(np.float32)


def colorize_by_height(points_xyz: np.ndarray):
    z = points_xyz[:, 2]
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    denom = (zmax - zmin) if (zmax - zmin) > 1e-9 else 1.0
    t = (z - zmin) / denom
    return np.column_stack([
        np.clip(1.5 * t, 0, 1),
        np.clip(1.5 * (1 - np.abs(t - 0.5) * 2), 0, 1),
        np.clip(1.5 * (1 - t), 0, 1),
    ]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--center", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel downsample in meters (0 disables)")
    args = ap.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        lon, lat = args.center
        bbox = bbox_from_center(lon, lat, args.radius_km)

    os.makedirs(args.out_dir, exist_ok=True)
    dem_tif = os.path.join(args.out_dir, "dem.tif")
    pcd_path = os.path.join(args.out_dir, "dem.pcd")
    ply_path = os.path.join(args.out_dir, "dem.ply")

    print("AOI bbox (EPSG:4326):", bbox)

    # 1) Find tiles
    print("Searching DEM tiles from Planetary Computer STAC...")
    items = search_dem_tiles(bbox)
    print("Tiles found:", len(items))

    # 2) Download + mosaic + clip -> dem.tif
    print("Downloading/mosaicking/clipping ->", dem_tif)
    mosaic_and_clip(items, bbox, dem_tif, cache_dir=os.path.join(args.out_dir, "_tiles"))
    print("Saved DEM:", dem_tif)

    # 3) Reproject to UTM + make point cloud
    print("Reprojecting to UTM (meters) and building point cloud...")
    dem_utm, tfm_utm, crs_utm, nodata = reproject_dem_to_utm(dem_tif)
    pts = dem_to_points(dem_utm, tfm_utm, nodata=nodata, stride=max(1, args.stride))
    print("Points:", len(pts), "UTM CRS:", crs_utm)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_by_height(pts).astype(np.float64))

    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)

    # 4) Visualize
    print("Visualizing (close the window to continue)...")
    o3d.visualization.draw_geometries([pcd], window_name="DEM Point Cloud")

    # 5) Save PCD/PLY
    o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=False)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
    print("Saved:", pcd_path)
    print("Saved:", ply_path)


if __name__ == "__main__":
    main()
