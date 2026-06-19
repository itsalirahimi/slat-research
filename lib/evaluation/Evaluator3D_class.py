#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import numpy as np
import open3d as o3d
import threading

Reg = o3d.pipelines.registration


class Evaluator3D:
    """
    3D point-cloud evaluator:
      - Global coarse init (scale-from-RMS + FGR/RANSAC)
      - Fast multi-scale Sim(3) ICP (trimmed, capped correspondences)
        with robust losses and optional point-to-plane weighting
        and optional multi-threaded KD search
      - Pose/scale errors + Shape metrics (RMSE, Chamfer², Hausdorff, Acc/Comp/F1)
      - Expanded metrics: surface coverage, density comparison, normal consistency, scale drift analysis
      - Human-readable analysis
      - Optional 2D depth evaluation (orthographic Z) with heatmaps
      - Saving colored aligned clouds (ref=red, test=blue) and combined
    """

    # ------------- ctor / config -------------
    def __init__(self,
                 visualize: bool = True,
                 cap_corr: int = 200_000,
                 metric_voxel_frac: float = 0.0,
                 kd_search_threads: int = 1,
                 robust_loss: Optional[str] = None,  # None | "huber" | "tukey"
                 robust_delta: float = 0.05):
        """
        Args:
            visualize: show Stage A/B/C windows (Open3D).
            cap_corr: max correspondences (sampled source points) per ICP iteration.
            metric_voxel_frac: optional voxel frac of bbox diag for metrics speedup (0=off).
            kd_search_threads: number of threads for NN search in ICP loops (>=1).
            robust_loss: apply robust loss to correspondences inside Sim(3) estimation ("huber" or "tukey").
            robust_delta: robust loss scale parameter (in meters).
        """
        self.visualize = visualize
        self.cap_corr = int(cap_corr)
        self.metric_voxel_frac = float(metric_voxel_frac)
        self.kd_search_threads = max(1, int(kd_search_threads))
        self.robust_loss = robust_loss
        self.robust_delta = float(robust_delta)

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
        # Solve Y ≈ s R X + t (unweighted)
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

    @staticmethod
    def robust_weights(residuals: np.ndarray, loss: Optional[str], delta: float) -> np.ndarray:
        if loss is None:
            return np.ones_like(residuals)
        r = residuals / max(delta, 1e-12)
        if loss.lower() == "huber":
            w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.abs(r))
        elif loss.lower() == "tukey":
            # Tukey biweight: w = (1 - (r^2))^2 for |r|<1, else 0
            w = np.zeros_like(r)
            m = np.abs(r) < 1.0
            rr = r[m]
            w[m] = (1.0 - rr * rr) ** 2
        else:
            w = np.ones_like(residuals)
        return w

    @staticmethod
    def umeyama_weighted(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        # Solve Y ≈ s R X + t with weights w (>=0)
        eps = 1e-15
        w = w.astype(np.float64)
        wsum = float(np.sum(w)) + eps
        mx = (w[:, None] * X).sum(axis=0) / wsum
        my = (w[:, None] * Y).sum(axis=0) / wsum
        Xc = X - mx
        Yc = Y - my
        S = (Yc.T @ (w[:, None] * Xc)) / wsum
        U, D, Vt = np.linalg.svd(S)
        Z = np.eye(3)
        Z[2, 2] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
        R = U @ Z @ Vt
        varx = np.sum(w * np.sum(Xc * Xc, axis=1)) / wsum + eps
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

    # ------------- KD search helpers (optional multithreading) -------------
    @staticmethod
    def _nn_search_batch(kdt: o3d.geometry.KDTreeFlann, Xsub: np.ndarray, out_idxs: np.ndarray, out_d2s: np.ndarray,
                         start: int, end: int):
        for i in range(start, end):
            p = Xsub[i]
            k, ind, d2 = kdt.search_knn_vector_3d(p, 1)
            if k == 1:
                out_idxs[i] = ind[0]; out_d2s[i] = d2[0]
            else:
                out_idxs[i] = -1; out_d2s[i] = np.inf

    def nn_search(self, kdt: o3d.geometry.KDTreeFlann, Xsub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = Xsub.shape[0]
        idxs = np.empty(n, np.int64)
        d2s = np.empty(n, np.float64)
        if self.kd_search_threads <= 1 or n < 2000:
            self._nn_search_batch(kdt, Xsub, idxs, d2s, 0, n)
            return idxs, d2s
        # multithreaded split
        threads = []
        chunk = max(1, n // self.kd_search_threads)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            th = threading.Thread(target=self._nn_search_batch, args=(kdt, Xsub, idxs, d2s, start, end))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        return idxs, d2s

    # ------------- fast multi-scale Sim(3) ICP -------------
    def sim3_icp_pyramid(self,
                         test_np: np.ndarray,
                         ref: o3d.geometry.PointCloud,
                         s0: float, R0: np.ndarray, t0: np.ndarray,
                         max_corr_fracs: Tuple[float, float, float] = (0.20, 0.10, 0.05),
                         trims: Tuple[float, float, float] = (0.50, 0.35, 0.20),
                         iters: Tuple[int, int, int] = (15, 10, 10), vis: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        s, R, t = float(s0), R0.copy(), t0.copy()
        d = self.bbox_diag(ref)
        ref_levels = [self.downsample_frac(ref, f) for f in (0.08, 0.04, 0.02)]

        for lvl, (ref_l, mc_frac, trim_q, niter) in enumerate(zip(ref_levels, max_corr_fracs, trims, iters)):
            max_corr = max(1e-4, mc_frac * d)
            if vis:
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

                # multi-threaded nearest neighbor search
                idxs, d2s = self.nn_search(kdt, Xsub)

                keep = np.isfinite(d2s) & (d2s <= max_corr * max_corr)
                if keep.sum() < 3:
                    break

                dists = np.sqrt(d2s[keep])
                if 0.0 < trim_q < 0.9 and keep.sum() > 20:
                    q = np.quantile(dists, 1.0 - trim_q)
                    K = np.where(keep)[0][dists <= q]
                    dists_kept = dists[dists <= q]
                else:
                    K = np.where(keep)[0]
                    dists_kept = dists

                src = Psub[K]
                dst = np.asarray(ref_l.points)[idxs[K]]

                # robust weights based on residual magnitudes
                w_r = self.robust_weights(dists_kept, self.robust_loss, delta=max(self.robust_delta, 1e-6))

                w = np.clip(w_r, 1e-6, 1.0)

                # Weighted Umeyama for Sim(3)
                s_new, R_new, t_new = self.umeyama_weighted(src, dst, w)

                ds = abs(s_new - s); dR = self.rot_angle_deg(R_new @ R.T); dt = float(np.linalg.norm(t_new - t))
                s, R, t = s_new, R_new, t_new
                if ds < 1e-7 and dR < 1e-4 and dt < 1e-5:
                    break

            if vis:
                self.show(f"Pyramid Level {lvl} (post-ICP)", ref_l,
                          self.np_to_o3d(self.transform_pts(test_np, s, R, t)))

        return s, R, t

    # ------------- metrics (base) -------------
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

    # ------------- metrics expansion -------------
    @staticmethod
    def surface_coverage_ratio(ref: o3d.geometry.PointCloud, test: o3d.geometry.PointCloud, tau: float) -> float:
        # fraction of ref points covered by test within tau
        d_ref_to_test = np.asarray(ref.compute_point_cloud_distance(test), float)
        if d_ref_to_test.size == 0:
            return 0.0
        return float(np.mean(d_ref_to_test <= tau))

    @staticmethod
    def density_comparison(ref: o3d.geometry.PointCloud, test: o3d.geometry.PointCloud,
                           voxel_size: float) -> Dict:
        # Compare occupancy counts in a shared voxel grid over XY bounds
        ref_min, ref_max = Evaluator3D.bbox_minmax(ref)
        test_min, test_max = Evaluator3D.bbox_minmax(test)
        mn = np.minimum(ref_min, test_min)
        mx = np.maximum(ref_max, test_max)

        def voxelize(pcd):
            P = np.asarray(pcd.points, float)
            if P.size == 0:
                return set()
            idx = np.floor((P - mn) / max(voxel_size, 1e-12)).astype(np.int64)
            # Use tuple hash of voxel indices
            return set(map(tuple, idx))
        v_ref = voxelize(ref)
        v_test = voxelize(test)

        inter = len(v_ref & v_test)
        union = len(v_ref | v_test)
        jacc = inter / max(union, 1e-12)
        cov_ref = inter / max(len(v_ref), 1e-12)
        cov_test = inter / max(len(v_test), 1e-12)
        return {"shared_voxels": inter, "union_voxels": union, "jaccard": jacc,
                "coverage_ref": cov_ref, "coverage_test": cov_test}

    @staticmethod
    def normal_consistency(ref: o3d.geometry.PointCloud, test: o3d.geometry.PointCloud, tau: float) -> Dict:
        # Requires normals; computes angular agreement at correspondences (within tau)
        if not ref.has_normals() or not test.has_normals():
            return {"available": False}
        d_ref_to_test = np.asarray(ref.compute_point_cloud_distance(test), float)
        P_ref = np.asarray(ref.points, float)
        P_test = np.asarray(test.points, float)
        N_ref = np.asarray(ref.normals, float)
        N_test = np.asarray(test.normals, float)

        # Build KD tree on test for nearest neighbor indices
        kdt = o3d.geometry.KDTreeFlann(test)
        angles = []
        for i, p in enumerate(P_ref):
            k, ind, _ = kdt.search_knn_vector_3d(p, 1)
            if k == 1:
                j = ind[0]
                if d_ref_to_test[i] <= tau:
                    # angle between normals
                    nr = N_ref[i] / (np.linalg.norm(N_ref[i]) + 1e-12)
                    nt = N_test[j] / (np.linalg.norm(N_test[j]) + 1e-12)
                    cosang = np.clip(np.dot(nr, nt), -1.0, 1.0)
                    ang_deg = float(np.degrees(np.arccos(cosang)))
                    angles.append(ang_deg)
        if not angles:
            return {"available": True, "count": 0, "mean_deg": float('nan'), "p90_deg": float('nan')}
        arr = np.array(angles, float)
        return {"available": True, "count": int(arr.size), "mean_deg": float(arr.mean()),
                "p90_deg": float(np.quantile(arr, 0.90))}

    def scale_drift_analysis(self,
                             ref: o3d.geometry.PointCloud,
                             test_aligned: o3d.geometry.PointCloud,
                             grid_frac: float = 0.1,
                             tau: Optional[float] = None) -> Dict:
        # Partition space into voxels of size grid_frac * bbox_diag(ref), estimate local scale via weighted Umeyama
        diag = self.bbox_diag(ref)
        vs = max(1e-6, grid_frac * diag)
        ref_min, _ = self.bbox_minmax(ref)

        Pref = np.asarray(ref.points, float)
        Ptest = np.asarray(test_aligned.points, float)

        # Build KD for correspondences
        kdt = o3d.geometry.KDTreeFlann(ref)
        # Voxel ids for test points
        vox = np.floor((Ptest - ref_min) / vs).astype(np.int64)
        # Map voxels -> indices
        vox_dict: Dict[Tuple[int, int, int], List[int]] = {}
        for i, v in enumerate(vox):
            key = (int(v[0]), int(v[1]), int(v[2]))
            vox_dict.setdefault(key, []).append(i)

        local_scales = []
        for key, idxs in vox_dict.items():
            if len(idxs) < 20:
                continue
            src = Ptest[idxs]
            # find nearest in ref for each src
            dst = np.empty_like(src)
            dists = np.empty(src.shape[0], float)
            for i, p in enumerate(src):
                k, ind, d2 = kdt.search_knn_vector_3d(p, 1)
                if k == 1:
                    dst[i] = Pref[ind[0]]
                    dists[i] = math.sqrt(d2[0])
                else:
                    dists[i] = np.inf
            keep = np.isfinite(dists)
            if np.sum(keep) < 10:
                continue
            src_k = src[keep]; dst_k = dst[keep]
            if tau is not None:
                k2 = (np.sqrt((np.sum((src_k - dst_k) ** 2, axis=1))) <= tau)
                src_k, dst_k = src_k[k2], dst_k[k2]
            if src_k.shape[0] < 10:
                continue
            # weights favor closer pairs
            w = self.robust_weights(np.linalg.norm(src_k - dst_k, axis=1), "huber", delta=max(self.robust_delta, 1e-6))
            s_loc, _, _ = self.umeyama_weighted(src_k, dst_k, w)
            local_scales.append(s_loc)

        if not local_scales:
            return {"count": 0, "mean_s": float('nan'), "std_s": float('nan'),
                    "min_s": float('nan'), "max_s": float('nan')}
        arr = np.array(local_scales, float)
        return {"count": int(arr.size), "mean_s": float(arr.mean()), "std_s": float(arr.std()),
                "min_s": float(arr.min()), "max_s": float(arr.max())}

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
        """
        # --- BEGIN: pipeline up to metrics & analysis ---
        ref = self.load_pcd(ref_pcd)
        test = self.load_pcd(test_pcd)

        if self.visualize:
            self.show("Stage A: Raw (no alignment)", ref, test)

        s0, R0, t0, ginfo = self.coarse_align(test, ref)
        test_coarse = self.np_to_o3d(self.transform_pts(np.asarray(test.points), s0, R0, t0))
        if self.visualize:
            self.show(f"Stage B: After coarse ({ginfo['method']}, s0≈{s0:.3f})", ref, test_coarse)

        s, R, t = self.sim3_icp_pyramid(np.asarray(test.points, float), ref, s0, R0, t0, vis=False)
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
        # --- END: pipeline up to metrics & analysis ---

        # -------- explicit colored exports --------
        if save_combined is None and save_aligned is not None:
            save_combined = save_aligned

        ref_red  = o3d.geometry.PointCloud(ref_m);  self.paint_uniform(ref_red,  (1, 0, 0))  # red
        tst_blue = o3d.geometry.PointCloud(test_m); self.paint_uniform(tst_blue, (0, 0, 1))  # blue

        saved_paths = []
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

        # Expanded metrics after alignment
        tau_cov = 0.01 * d  # 1% of diag as coverage tolerance
        cov_ratio = self.surface_coverage_ratio(ref_m, test_m, tau_cov)
        dens = self.density_comparison(ref_m, test_m, voxel_size=0.02 * d)
        # ensure normals for normal consistency
        if not ref_m.has_normals():
            self.estimate_normals(ref_m, radius=max(1e-4, 0.02 * d))
        if not test_m.has_normals():
            self.estimate_normals(test_m, radius=max(1e-4, 0.02 * d))
        norm_cons = self.normal_consistency(ref_m, test_m, tau=tau_cov)
        scale_drift = self.scale_drift_analysis(ref_m, test_m, grid_frac=0.1, tau=tau_cov)

        # Package report (optionally write JSON)
        report = {
            "scale": float(s),
            "rotation_matrix": R.tolist(),
            "translation": t.tolist(),
            "scale_error_abs": float(scale_err),
            "rot_error_deg": float(rot_err),
            "trans_error_m": float(trans_err),
            "metrics": {
                "RMSE_XtoY_m": shape["RMSE_XtoY_m"],
                "Chamfer2_sym_m2": shape["Chamfer2_sym_m2"],
                "Hausdorff_sym_m": shape["Hausdorff_sym_m"],
                "by_threshold": shape["by_threshold"],
                "quantiles_test_to_ref": shape["quantiles_test_to_ref"],
                "quantiles_ref_to_test": shape["quantiles_ref_to_test"]
            },
            "expanded_metrics": {
                "surface_coverage_ratio": cov_ratio,
                "density_comparison": dens,
                "normal_consistency": norm_cons,
                "scale_drift": scale_drift
            },
            "settings": {
                "kd_search_threads": self.kd_search_threads,
                "robust_loss": self.robust_loss,
                "robust_delta": self.robust_delta
            }
        }
        if report_path is not None:
            try:
                import json
                rp = Path(report_path)
                rp.parent.mkdir(parents=True, exist_ok=True)
                with open(rp, "w") as f:
                    json.dump(report, f, indent=2)
                print("Report saved:", str(rp))
            except Exception as e:
                print("Report save failed:", e)

        return report


if __name__ == "__main__":
    # Minimal example run (edit paths as needed):
    ev = Evaluator3D(
        visualize=True,
        cap_corr=200_000,
        metric_voxel_frac=0.0,
        kd_search_threads=4,              # multi-threaded KD search
        robust_loss="huber",              # robust loss on correspondences
        robust_delta=0.02                 # meters (tune to your scene scale)
    )

    _ = ev.eval(
        ref_pcd="gt1.pcd",
        test_pcd="fused1.pcd",

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
        # , save_plots_dir="out/residual_plots"
    )
