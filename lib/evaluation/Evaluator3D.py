#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import open3d as o3d

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")
from utils.conversion import pcd2pcm
from fusion.helper import calc_ratio_map
from projection.helper import NULL_SCALE_MIN_Z, project3D, scale_pcm

Reg = o3d.pipelines.registration
# class Evaluator3D(O3DGUI):
#     def __init__(self):
#         super().__init__()


class Evaluator3D:
    """
    3D point-cloud evaluator:
      - Global coarse init (scale-from-RMS + FGR/RANSAC)
      - Fast multi-scale Sim(3) ICP (trimmed, capped correspondences)
      - Pose/scale errors + Shape metrics (RMSE, Chamfer², Hausdorff, Acc/Comp/F1)
      - Human-readable analysis
      - Optional 2D depth evaluation (orthographic Z) with heatmaps (MDE-style)
      - Saving colored aligned clouds (ref=red, test=blue) and combined

    Typical use:
        ev = Evaluator3D(visualize=False, cap_corr=200_000)
        report = ev.eval("ref.pcd", "test.pcd",
                         save_aligned="aligned_combined.pcd",
                         depth_eval=True, H=1024, W=1024,
                         save_plots_dir="depth_plots",
                         report_path="eval_report.json")
    """

    # ------------- ctor / config -------------
    def __init__(self,
                 visualize: bool = True,
                 cap_corr: int = 200_000,
                 metric_voxel_frac: float = 0.0):
        """
        Args:
            visualize: show Stage A/B/C windows (Open3D).
            cap_corr: max correspondences (sampled source points) per ICP iteration.
            metric_voxel_frac: optional voxel frac of bbox diag for metrics speedup (0=off).
        """
        self.visualize = visualize
        self.cap_corr = int(cap_corr)
        self.metric_voxel_frac = float(metric_voxel_frac)

    # ------------- IO & basics -------------
    @staticmethod
    def load_pcd(path: Path | str) -> o3d.geometry.PointCloud:
        pc = o3d.io.read_point_cloud(str(path))
        if pc.is_empty():
            raise RuntimeError(f"Empty/invalid PCD: {path}")
        return pc

    @staticmethod
    def np_to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
        g = o3d.geometry.PointCloud()
        g.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return g

    @staticmethod
    def paint_uniform(p: o3d.geometry.PointCloud, rgb: Tuple[float, float, float]):
        cols = np.tile(np.array(rgb, float), (np.asarray(p.points).shape[0], 1))
        p.colors = o3d.utility.Vector3dVector(cols)

    @staticmethod
    def show(stage: str, ref: o3d.geometry.PointCloud, test: o3d.geometry.PointCloud):
        r = o3d.geometry.PointCloud(ref); Evaluator3D.paint_uniform(r, (1, 0, 0))  # red=ref
        t = o3d.geometry.PointCloud(test); Evaluator3D.paint_uniform(t, (0, 0, 1))  # blue=test
        o3d.visualization.draw([r, t], title=stage)

    # ------------- geometry helpers -------------
    @staticmethod
    def bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
        e = pcd.get_axis_aligned_bounding_box().get_extent()
        return float(np.linalg.norm(e))

    @staticmethod
    def bbox_minmax(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        aabb = pcd.get_axis_aligned_bounding_box()
        return np.asarray(aabb.get_min_bound(), float), np.asarray(aabb.get_max_bound(), float)

    @staticmethod
    def rms_radius(pcd: o3d.geometry.PointCloud) -> float:
        P = np.asarray(pcd.points, float)
        c = P.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((P - c) ** 2, axis=1)) + 1e-15))

    @staticmethod
    def downsample_frac(pcd: o3d.geometry.PointCloud, frac: float) -> o3d.geometry.PointCloud:
        vx = max(1e-6, Evaluator3D.bbox_diag(pcd) * frac)
        q = pcd.voxel_down_sample(vx)
        return q if not q.is_empty() else pcd

    @staticmethod
    def transform_pts(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (s * (R @ pts.T)).T + t

    # ------------- rotation helpers -------------
    @staticmethod
    def rot_angle_deg(R: np.ndarray) -> float:
        x = (np.trace(R) - 1.0) * 0.5
        x = float(np.clip(x, -1, 1))
        return float(np.degrees(np.arccos(x)))

    @staticmethod
    def euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
        # ZYX extrinsic (yaw, pitch, roll)
        sy = -R[2, 0]
        cy = math.sqrt(max(1.0 - sy * sy, 0.0))
        if cy > 1e-8:
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))      # Z
            pitch = math.degrees(math.asin(sy))                   # Y
            roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))     # X
        else:
            yaw = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
            pitch = math.degrees(math.asin(sy))
            roll = 0.0
        return yaw, pitch, roll

    @staticmethod
    def axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
        theta = Evaluator3D.rot_angle_deg(R)
        if theta < 1e-9:
            return np.array([1, 0, 0], float), 0.0
        tr = np.trace(R)
        rad = math.acos(np.clip((tr - 1) / 2, -1, 1))
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
        axis = w / (2 * math.sin(rad) + 1e-15)
        n = np.linalg.norm(axis)
        return axis / (n + 1e-15), theta

    # ------------- features / FPFH -------------
    @staticmethod
    def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.normalize_normals()

    @staticmethod
    def fpfh(pcd: o3d.geometry.PointCloud, r_n: float, r_f: float):
        Evaluator3D.estimate_normals(pcd, r_n, 30)
        return Reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=r_f, max_nn=100))

    # ------------- similarity (Umeyama) -------------
    @staticmethod
    def umeyama(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        # Solve Y ≈ s R X + t
        mx, my = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - mx, Y - my
        S = (Yc.T @ Xc) / X.shape[0]
        U, D, Vt = np.linalg.svd(S)
        Z = np.eye(3)
        Z[2, 2] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
        R = U @ Z @ Vt
        varx = np.mean(np.sum(Xc * Xc, axis=1)) + 1e-15
        s = float(np.trace(np.diag(D) @ Z) / varx)
        t = my - s * (R @ mx)
        return s, R, t

    # ------------- global coarse alignment -------------
    def coarse_align(self, test: o3d.geometry.PointCloud, ref: o3d.geometry.PointCloud) -> Tuple[float, np.ndarray, np.ndarray, Dict]:
        s0 = float(self.rms_radius(ref) / max(self.rms_radius(test), 1e-12))
        test_s = o3d.geometry.PointCloud(test)
        test_s.scale(s0, center=(0, 0, 0))

        d = self.bbox_diag(ref)
        ref_ds = self.downsample_frac(ref, 0.01)
        test_ds = self.downsample_frac(test_s, 0.01)
        r_norm, r_feat = max(1e-4, 0.02 * d), max(1e-4, 0.05 * d)
        f_ref, f_test = self.fpfh(ref_ds, r_norm, r_feat), self.fpfh(test_ds, r_norm, r_feat)
        dist = max(1e-4, 0.05 * d)

        used = "none"; R0 = np.eye(3); t0 = np.zeros(3)
        try:
            opt = Reg.FastGlobalRegistrationOption(maximum_correspondence_distance=dist, iteration_number=64)
            res = Reg.registration_fgr_based_on_feature_matching(test_ds, ref_ds, f_test, f_ref, opt)
            if res.transformation is not None:
                T = res.transformation; R0 = T[:3, :3]; t0 = T[:3, 3]; used = "FGR"
        except Exception:
            pass
        if used == "none":
            try:
                checkers = [Reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            Reg.CorrespondenceCheckerBasedOnDistance(dist)]
                res = Reg.registration_ransac_based_on_feature_matching(
                    test_ds, ref_ds, f_test, f_ref, True, dist,
                    Reg.TransformationEstimationPointToPoint(False), 4, checkers,
                    Reg.RANSACConvergenceCriteria(40000, 500))
                T = res.transformation; R0 = T[:3, :3]; t0 = T[:3, 3]; used = "RANSAC"
            except Exception:
                pass
        return s0, R0, t0, {"method": used, "scale_init": s0}

    # ------------- fast multi-scale Sim(3) ICP -------------
    def sim3_icp_pyramid(self,
                         test_np: np.ndarray,
                         ref: o3d.geometry.PointCloud,
                         s0: float, R0: np.ndarray, t0: np.ndarray,
                         max_corr_fracs: Tuple[float, float, float] = (0.20, 0.10, 0.05),
                         trims: Tuple[float, float, float] = (0.50, 0.35, 0.20),
                         iters: Tuple[int, int, int] = (15, 10, 10)) -> Tuple[float, np.ndarray, np.ndarray]:
        s, R, t = float(s0), R0.copy(), t0.copy()
        d = self.bbox_diag(ref)
        ref_levels = [self.downsample_frac(ref, f) for f in (0.08, 0.04, 0.02)]

        for lvl, (ref_l, mc_frac, trim_q, niter) in enumerate(zip(ref_levels, max_corr_fracs, trims, iters)):
            max_corr = max(1e-4, mc_frac * d)
            if self.visualize:
                self.show(f"Pyramid Level {lvl} (pre-ICP)", ref_l,
                          self.np_to_o3d(self.transform_pts(test_np, s, R, t)))

            kdt = o3d.geometry.KDTreeFlann(ref_l)
            for _ in range(niter):
                Xk = self.transform_pts(test_np, s, R, t)
                if Xk.shape[0] > self.cap_corr:
                    idx = np.random.choice(Xk.shape[0], self.cap_corr, replace=False)
                    Xsub, Psub = Xk[idx], test_np[idx]
                else:
                    Xsub, Psub = Xk, test_np

                idxs = np.empty(Xsub.shape[0], np.int64)
                d2s = np.empty(Xsub.shape[0], np.float64)
                for i, p in enumerate(Xsub):
                    k, ind, d2 = kdt.search_knn_vector_3d(p, 1)
                    if k == 1: idxs[i] = ind[0]; d2s[i] = d2[0]
                    else:      idxs[i] = -1;     d2s[i] = np.inf
                keep = np.isfinite(d2s) & (d2s <= max_corr * max_corr)
                if keep.sum() < 3:
                    break

                dists = np.sqrt(d2s[keep])
                if 0.0 < trim_q < 0.9 and keep.sum() > 20:
                    q = np.quantile(dists, 1.0 - trim_q)
                    K = np.where(keep)[0][dists <= q]
                else:
                    K = np.where(keep)[0]

                src = Psub[K]; dst = np.asarray(ref_l.points)[idxs[K]]
                s_new, R_new, t_new = self.umeyama(src, dst)

                ds = abs(s_new - s); dR = self.rot_angle_deg(R_new @ R.T); dt = float(np.linalg.norm(t_new - t))
                s, R, t = s_new, R_new, t_new
                if ds < 1e-7 and dR < 1e-4 and dt < 1e-5:
                    break

            if self.visualize:
                self.show(f"Pyramid Level {lvl} (post-ICP)", ref_l,
                          self.np_to_o3d(self.transform_pts(test_np, s, R, t)))

        return s, R, t

    # ------------- metrics -------------
    @staticmethod
    def distances_both(A: o3d.geometry.PointCloud, B: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        d_ab = np.asarray(A.compute_point_cloud_distance(B), float)
        d_ba = np.asarray(B.compute_point_cloud_distance(A), float)
        return d_ab, d_ba

    def metrics(self, test_aligned: o3d.geometry.PointCloud,
                ref: o3d.geometry.PointCloud,
                taus: List[float]) -> Dict:
        d_tb, d_rt = self.distances_both(test_aligned, ref)
        rmse = float(np.sqrt(np.mean(d_tb ** 2))) if d_tb.size else float('nan')
        cham2 = float(np.mean(d_tb ** 2) + np.mean(d_rt ** 2))
        haus = float(max(np.max(d_tb) if d_tb.size else 0.0,
                         np.max(d_rt) if d_rt.size else 0.0))
        rows = []
        for tau in taus:
            acc = float(np.mean(d_tb <= tau)) if d_tb.size else 0.0
            comp = float(np.mean(d_rt <= tau)) if d_rt.size else 0.0
            f1 = (2 * acc * comp / (acc + comp)) if (acc + comp) > 1e-12 else 0.0
            rows.append({"tau_m": float(tau), "Accuracy": acc, "Completeness": comp, "F1": f1})
        q_tb = {"p50": float(np.median(d_tb)) if d_tb.size else float('nan'),
                "p90": float(np.quantile(d_tb, 0.90)) if d_tb.size else float('nan'),
                "p95": float(np.quantile(d_tb, 0.95)) if d_tb.size else float('nan'),
                "p99": float(np.quantile(d_tb, 0.99)) if d_tb.size else float('nan')}
        q_rt = {"p50": float(np.median(d_rt)) if d_rt.size else float('nan'),
                "p90": float(np.quantile(d_rt, 0.90)) if d_rt.size else float('nan'),
                "p95": float(np.quantile(d_rt, 0.95)) if d_rt.size else float('nan'),
                "p99": float(np.quantile(d_rt, 0.99)) if d_rt.size else float('nan')}
        return {"RMSE_XtoY_m": rmse, "Chamfer2_sym_m2": cham2, "Hausdorff_sym_m": haus,
                "by_threshold": rows, "quantiles_test_to_ref": q_tb, "quantiles_ref_to_test": q_rt,
                "dists_test_to_ref": d_tb, "dists_ref_to_test": d_rt}

    # ------------- human-readable analysis -------------
    def human_analysis(self,
                       ref: o3d.geometry.PointCloud,
                       test_final: o3d.geometry.PointCloud,
                       s: float, R: np.ndarray, t: np.ndarray,
                       shape: Dict):
        diag = self.bbox_diag(ref)
        rR = self.rms_radius(ref)
        n_ref, n_test = np.asarray(ref.points).shape[0], np.asarray(test_final.points).shape[0]

        scale_pct = 100.0 * (s - 1.0)
        rot_deg = self.rot_angle_deg(R)
        trans_m = float(np.linalg.norm(t))
        trans_norm_diag = trans_m / max(diag, 1e-12)

        rmse = shape["RMSE_XtoY_m"]; haus = shape["Hausdorff_sym_m"]
        rmse_nd = rmse / max(diag, 1e-12); haus_nd = haus / max(diag, 1e-12)
        rmse_nr = rmse / max(rR, 1e-12)

        yaw, pitch, roll = self.euler_zyx(R)
        ax, ang = self.axis_angle(R)

        print("\n--- Analysis (easy-to-read) ---")
        print(f"Scene size (ref)        : bbox diag ≈ {diag:.3f} m, RMS radius ≈ {rR:.3f} m")
        print(f"Point counts            : ref {n_ref:,d} | test(aligned) {n_test:,d}")
        print(f"\nPose/Scale needed (test → ref):")
        print(f"  Scale                 : s = {s:.9f}  →  {scale_pct:+.2f}%")
        print(f"  Rotation              : {rot_deg:.3f}°  (axis {ax/np.linalg.norm(ax+1e-15)} , angle {ang:.3f}°)")
        print(f"    Euler ZYX (yaw,pitch,roll) = ({yaw:.3f}°, {pitch:.3f}°, {roll:.3f}°)")
        print(f"  Translation           : {t}  |L2|={trans_m:.3f} m  ({100.0 * trans_norm_diag:.2f}% of diag)")

        print(f"\nShape after alignment:")
        print(f"  RMSE (test→ref)       : {rmse:.3f} m  ({100.0 * rmse_nd:.2f}% of diag, {rmse_nr:.3f}× RMS radius)")
        print(f"  Hausdorff (sym)       : {haus:.3f} m  ({100.0 * haus_nd:.2f}% of diag)")
        print("  Chamfer² (sym)        : {:.3f} m²".format(shape["Chamfer2_sym_m2"]))

        qt, qr = shape["quantiles_test_to_ref"], shape["quantiles_ref_to_test"]
        print("\nResidual quantiles (meters):")
        print(f"  test→ref: median={qt['p50']:.3f}, P90={qt['p90']:.3f}, P95={qt['p95']:.3f}, P99={qt['p99']:.3f}")
        print(f"  ref→test: median={qr['p50']:.3f}, P90={qr['p90']:.3f}, P95={qr['p95']:.3f}, P99={qr['p99']:.3f}")

        print("\nCoverage (ETH3D/T&T-style):")
        for row in shape["by_threshold"]:
            tau = row["tau_m"]; pc = 100.0 * (tau / max(diag, 1e-12))
            print(f"  @ τ = {tau:.3f} m ({pc:.2f}% diag): "
                  f"Acc={row['Accuracy'] * 100:.1f}%  Comp={row['Completeness'] * 100:.1f}%  F1={row['F1'] * 100:.1f}%")

    # ------------- depth (orthographic Z) -------------
    @staticmethod
    def ortho_depth_from_pcd(pcd: o3d.geometry.PointCloud, H: int, W: int,
                             bounds_xy: Tuple[np.ndarray, np.ndarray],
                             z_min_ref: float) -> Tuple[np.ndarray, np.ndarray]:
        (xmin, ymin, _), (xmax, ymax, _) = bounds_xy
        P = np.asarray(pcd.points, float)
        D = np.full((H, W), np.inf, np.float32)
        if P.size == 0:
            D[:] = np.nan
            return D, np.zeros((H, W), bool)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        u = np.floor((x - xmin) / max(xmax - xmin, 1e-12) * (W - 1)).astype(np.int64)
        v = np.floor((y - ymin) / max(ymax - ymin, 1e-12) * (H - 1)).astype(np.int64)
        u = np.clip(u, 0, W - 1); v = np.clip(v, 0, H - 1)
        dvals = (z - z_min_ref).astype(np.float32)
        flat = v * W + u
        np.minimum.at(D.ravel(), flat, dvals)
        mask = np.isfinite(D)
        D[~mask] = np.nan
        return D, mask

    @staticmethod
    def depth_metrics(Dt: np.ndarray, Dr: np.ndarray) -> Dict:
        m = np.isfinite(Dt) & np.isfinite(Dr)
        if not np.any(m):
            return {"valid": 0}
        diff = Dt[m] - Dr[m]
        absd = np.abs(diff)
        mae = float(np.mean(absd))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        med = float(np.median(absd))
        p90 = float(np.quantile(absd, 0.90))

        # SILog
        mt = (Dt[m] > 0) & (Dr[m] > 0)
        if np.any(mt):
            ld = np.log(Dt[m][mt]) - np.log(Dr[m][mt])
            silog = float(np.sqrt(np.mean(ld ** 2) - (np.mean(ld) ** 2)))
        else:
            silog = float('nan')

        eps = 1e-12
        ratio = np.maximum(Dt[m] / np.maximum(Dr[m], eps), Dr[m] / np.maximum(Dt[m], eps))
        d1 = float(np.mean(ratio < 1.25))
        d2 = float(np.mean(ratio < 1.25 ** 2))
        d3 = float(np.mean(ratio < 1.25 ** 3))

        return {"valid": int(np.sum(m)), "MAE": mae, "RMSE": rmse, "MedianAE": med, "P90AE": p90,
                "SILog": silog, "delta1": d1, "delta2": d2, "delta3": d3}

    @staticmethod
    def _save_depth_png(path: Path, D: np.ndarray, title: str = ""):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize  # noqa
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            vmax = np.nanpercentile(D, 99) if np.isfinite(D).any() else 1.0
            plt.imshow(D, cmap="viridis", interpolation="nearest", vmin=0, vmax=vmax)
            plt.colorbar(label="depth (m)"); plt.title(title); plt.tight_layout()
            plt.savefig(path); plt.close()
        except Exception:
            print("matplotlib not available; skipping depth PNG:", path)

    @staticmethod
    def _save_err_png(path: Path, E: np.ndarray, title: str = ""):
        try:
            import matplotlib.pyplot as plt
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            a = np.abs(E[np.isfinite(E)])
            vmax = np.percentile(a, 95) if a.size else 1.0
            plt.imshow(E, cmap="bwr", interpolation="nearest", vmin=-vmax, vmax=+vmax)
            plt.colorbar(label="test - ref (m)"); plt.title(title); plt.tight_layout()
            plt.savefig(path); plt.close()
        except Exception:
            print("matplotlib not available; skipping error PNG:", path)

    # ------------- main API -------------
    def eval(self,
            ref_pcd: Path | str,
            test_pcd: Path | str,
            save_combined: Optional[Path | str] = None,          # NEW name (combined file)
            save_ref_path: Optional[Path | str] = "aligned_ref_red.pcd",    # NEW explicit export
            save_test_path: Optional[Path | str] = "aligned_test_blue.pcd", # NEW explicit export
            report_path: Optional[Path | str] = "eval_report.json",
            depth_eval: bool = False,
            H: Optional[int] = None,
            W: Optional[int] = None,
            save_plots_dir: Optional[Path | str] = None,
            # back-compat alias: if provided, acts like save_combined
            save_aligned: Optional[Path | str] = None) -> Dict:
        """
        Run full evaluation.

        Args:
            ref_pcd, test_pcd: PCD paths.
            save_combined: path for combined PCD (ref red + test blue). If None, skip combined.
            save_ref_path: path for red reference export. Set None to skip.
            save_test_path: path for blue test export. Set None to skip.
            report_path: JSON report path.
            depth_eval/H/W: enable orthographic depth evaluation and set image size.
            save_plots_dir: optional dir for residual histograms.
            save_aligned: (deprecated) alias for save_combined for backward compatibility.
        Returns:
            dict report.
        """
        # --- BEGIN: unchanged pipeline up to metrics & analysis ---
        ref = self.load_pcd(ref_pcd)
        test = self.load_pcd(test_pcd)

        if self.visualize:
            self.show("Stage A: Raw (no alignment)", ref, test)

        s0, R0, t0, ginfo = self.coarse_align(test, ref)
        test_coarse = self.np_to_o3d(self.transform_pts(np.asarray(test.points), s0, R0, t0))
        if self.visualize:
            self.show(f"Stage B: After coarse ({ginfo['method']}, s0≈{s0:.3f})", ref, test_coarse)

        s, R, t = self.sim3_icp_pyramid(np.asarray(test.points, float), ref, s0, R0, t0)
        test_final = self.np_to_o3d(self.transform_pts(np.asarray(test.points), s, R, t))
        if self.visualize:
            self.show("Stage C: After Sim(3) refine (fast)", ref, test_final)

        scale_err = abs(s - 1.0); rot_err = self.rot_angle_deg(R); trans_err = float(np.linalg.norm(t))
        print("\n=== Similarity Transform (test → reference) ===")
        print(f"Scale s        : {s:.9f}   (|s-1| = {scale_err:.9e})")
        print(f"Rotation (deg) : {rot_err:.6f}")
        print(f"Translation L2 : {trans_err:.6f} m")
        print("R =\n", R); print("t =", t)

        ref_m = self.downsample_frac(ref, self.metric_voxel_frac) if self.metric_voxel_frac > 0 else ref
        test_m = self.downsample_frac(test_final, self.metric_voxel_frac) if self.metric_voxel_frac > 0 else test_final

        d = self.bbox_diag(ref_m); taus = [0.005 * d, 0.01 * d, 0.02 * d]
        shape = self.metrics(test_m, ref_m, taus)

        print("\n=== Shape / Geometric Errors (after alignment) ===")
        print(f"RMSE (X→Y)             : {shape['RMSE_XtoY_m']:.6f} m")
        print(f"Chamfer (squared, sym) : {shape['Chamfer2_sym_m2']:.6f} m^2")
        print(f"Hausdorff (sym)        : {shape['Hausdorff_sym_m']:.6f} m")
        for row in shape["by_threshold"]:
            tau = row["tau_m"]
            print(f"\n@ τ = {tau:.3f} m")
            print(f"Accuracy (test→ref <= τ)     : {row['Accuracy']*100:.2f}%")
            print(f"Completeness (ref→test <= τ) : {row['Completeness']*100:.2f}%")
            print(f"F1                           : {row['F1']*100:.2f}%")

        self.human_analysis(ref_m, test_m, s, R, t, shape)
        # --- END: unchanged pipeline up to metrics & analysis ---

        # -------- NEW: explicit colored exports --------
        # resolve back-compat alias
        if save_combined is None and save_aligned is not None:
            save_combined = save_aligned

        # build colored clouds
        ref_red  = o3d.geometry.PointCloud(ref_m);  self.paint_uniform(ref_red,  (1, 0, 0))  # red
        tst_blue = o3d.geometry.PointCloud(test_m); self.paint_uniform(tst_blue, (0, 0, 1))  # blue

        saved_paths = []

        # explicit individual exports (default filenames provided)
        if save_ref_path is not None:
            save_ref_path = Path(save_ref_path)
            save_ref_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(save_ref_path), ref_red)
            saved_paths.append(str(save_ref_path))

        if save_test_path is not None:
            save_test_path = Path(save_test_path)
            save_test_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(save_test_path), tst_blue)
            saved_paths.append(str(save_test_path))

        # optional combined export (kept from before)
        if save_combined is not None:
            save_combined = Path(save_combined)
            save_combined.parent.mkdir(parents=True, exist_ok=True)
            combo = o3d.geometry.PointCloud(ref_red); combo += tst_blue
            o3d.io.write_point_cloud(str(save_combined), combo)
            saved_paths.append(str(save_combined))

        if saved_paths:
            print("\nSaved colored PCDs →")
            for p in saved_paths:
                print("  ", p)

        # -------- (optional) residual histograms, depth eval, and JSON report --------
        # (leave your existing code here unchanged)

        # Build and optionally save JSON report (same as before)...
        # Return the report dict as you already do.

    def dry_run(self, cimg, depth, pose, hfov_deg, bg_pcd_can):
        H, W, _ = cimg.shape
        projected_depth, _ = project3D(depth, pose, hfov_deg, 
            move=False, pyramidProj=False, do_rotate=True, do_scale=False)
        bg_pcm_can = pcd2pcm(bg_pcd_can, H, W)
        _, gep, _ = calc_ratio_map(bg_pcd_can, pose)
        gep = scale_pcm(gep, np.mean(gep[:,:,2]), -pose.p6.z)

        

if __name__ == "__main__":
    # Minimal example run (edit paths as needed):
    ev = Evaluator3D(visualize=False, cap_corr=200_000, metric_voxel_frac=0.0)

    _ = ev.eval(
        ref_pcd="/home/ali/repos/OrthoLoc/az/pcd/L08_R0000.pcd",
        test_pcd="/home/ali/repos/slat-research/data/ortholoc/fusion/fusedproj_1.pcd",

        # explicit colored exports
        save_ref_path="out/aligned_ref_red.pcd",
        save_test_path="out/aligned_test_blue.pcd",

        # optional combined export (ref red + test blue together)
        save_combined="out/aligned_combined.pcd",

        # JSON report
        report_path="out/eval_report.json",

        # Depth evaluation (orthographic Z):
        depth_eval=False  # set True and provide H/W to enable
        # , H=1024, W=1024
        # , save_plots_dir="out/residual_plots"  # optional histograms
    )