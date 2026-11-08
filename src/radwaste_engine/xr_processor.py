import os
import numpy as np
import SimpleITK as sitk
import xarray as xr

from src.config import load_config



def process_ct(
    ct_filepath,
    th_air=2.47,
    th_filler=59,
    export_fractions=True,
    out_dir="ct_products",
    remove_barrel=None,
    barrel_mask_params=None,
    config=None,
):
    """
    production version with selectable downsampling method

    mesh.values:
        COARSE Mean-like CT (x,y,z)

    Available methods:
        - baseline                -> linear resample to coarse grid
        - bspline                 -> B-spline resample to coarse grid
        - oversample_block_average -> linear resample to aligned fine grid + block average

    Optional:
        - barrel removal in hi-res before downsampling
        - export coarse fractions from hi-res thresholding
    """

    # Resolve CT settings once. Explicit legacy arguments still take precedence.
    # Keep fraction exports intact until the CT2LAC/BYPASS cleanup is integrated.
    settings = (config if config is not None else load_config())["ct_processing"]
    ds_method = settings["downsampling_method"]
    oversample_factor = int(settings["oversample_factor"])
    final_size_vox = tuple(int(v) for v in settings["grid_size"])
    final_spacing_mm = tuple(float(v) for v in settings["grid_spacing_mm"])
    shift_mm = tuple(float(v) for v in settings["shift_mm"])
    barrel_settings = settings["barrel_removal"]
    if remove_barrel is None:
        remove_barrel = bool(barrel_settings["enabled"])
    if barrel_mask_params is None:
        barrel_mask_params = {
            "cx_mm": float(barrel_settings["center_x_mm"]),
            "cz_mm": float(barrel_settings["center_z_mm"]),
            "r_in_mm": float(barrel_settings["inner_radius_mm"]),
            "r_out_mm": float(barrel_settings["outer_radius_mm"]),
            "air_value": float(barrel_settings["replacement_value"]),
        }

    # ============================================================
    # Helpers: barrel removal
    # ============================================================
    def _make_cylindrical_shell_mask_xyz(arr_xyz, img, cx_mm, cz_mm, r_in_mm, r_out_mm,
                                         ymin_mm=None, ymax_mm=None):
        origin = np.array(img.GetOrigin(), dtype=float)
        spacing = np.array(img.GetSpacing(), dtype=float)
        direction = np.array(img.GetDirection(), dtype=float)

        if not np.allclose(direction, np.eye(3).ravel()):
            print("[WARNING] Non-identity direction. Assuming axis-aligned geometry.")

        nx, ny, nz = arr_xyz.shape

        xs_mm = origin[0] + np.arange(nx) * spacing[0]
        ys_mm = origin[1] + np.arange(ny) * spacing[1]
        zs_mm = origin[2] + np.arange(nz) * spacing[2]

        # barrel axis along y -> circular section in x-z
        X, Z = np.meshgrid(xs_mm, zs_mm, indexing="ij")
        R = np.sqrt((X - cx_mm) ** 2 + (Z - cz_mm) ** 2)

        shell_xz = (R >= r_in_mm) & (R <= r_out_mm)

        if ymin_mm is None:
            ymin_mm = float(ys_mm.min())
        if ymax_mm is None:
            ymax_mm = float(ys_mm.max())

        y_mask = (ys_mm >= ymin_mm) & (ys_mm <= ymax_mm)

        mask = np.zeros_like(arr_xyz, dtype=bool)
        for iy in range(ny):
            if y_mask[iy]:
                mask[:, iy, :] = shell_xz

        return mask

    def _apply_barrel_removal_to_image(ct_img, cx_mm, cz_mm, r_in_mm, r_out_mm,
                                       ymin_mm=None, ymax_mm=None, air_value=0.0):
        arr_zyx = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (z,y,x)
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))                   # (x,y,z)

        mask_xyz = _make_cylindrical_shell_mask_xyz(
            arr_xyz, ct_img,
            cx_mm=cx_mm,
            cz_mm=cz_mm,
            r_in_mm=r_in_mm,
            r_out_mm=r_out_mm,
            ymin_mm=ymin_mm,
            ymax_mm=ymax_mm,
        )

        arr_xyz_mod = arr_xyz.copy()
        arr_xyz_mod[mask_xyz] = float(air_value)

        arr_zyx_mod = np.transpose(arr_xyz_mod, (2, 1, 0))
        out_img = sitk.GetImageFromArray(arr_zyx_mod.astype(np.float32))
        out_img.CopyInformation(ct_img)

        return out_img, mask_xyz

    # ============================================================
    # Input CT
    # ============================================================
    ct_image = sitk.ReadImage(ct_filepath)


    barrel_removed_voxels = None
    if remove_barrel:
        ct_image, mask_xyz = _apply_barrel_removal_to_image(
            ct_image,
            cx_mm=barrel_mask_params["cx_mm"],
            cz_mm=barrel_mask_params["cz_mm"],
            r_in_mm=barrel_mask_params["r_in_mm"],
            r_out_mm=barrel_mask_params["r_out_mm"],
            ymin_mm=barrel_mask_params.get("ymin_mm", None),
            ymax_mm=barrel_mask_params.get("ymax_mm", None),
            air_value=barrel_mask_params.get("air_value", 0.0),
        )
        barrel_removed_voxels = int(mask_xyz.sum())
        print(f"[process_ct] Barrel removal active. Masked voxels: {barrel_removed_voxels}")

    # ============================================================
    # Fixed coarse grid
    # ============================================================
    final_origin = [-(0.5 * (n - 1) * s) for n, s in zip(final_size_vox, final_spacing_mm)]
    final_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)
    translation = sitk.TranslationTransform(3, shift_mm)

    # ============================================================
    # Downsampling choice
    # ============================================================
    if ds_method == "baseline":
        ct_coarse = sitk.Resample(
            ct_f,
            final_size_vox,
            translation,
            sitk.sitkLinear,
            final_origin,
            final_spacing_mm,
            final_direction,
            0.0,
            sitk.sitkFloat32,
        )

        arr_zyx = sitk.GetArrayFromImage(ct_coarse).astype(np.float32)
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
        ds_method_used = "baseline"

    elif ds_method == "bspline":
        ct_coarse = sitk.Resample(
            ct_f,
            final_size_vox,
            translation,
            sitk.sitkBSpline,
            final_origin,
            final_spacing_mm,
            final_direction,
            0.0,
            sitk.sitkFloat32,
        )

        arr_zyx = sitk.GetArrayFromImage(ct_coarse).astype(np.float32)
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
        ds_method_used = "bspline"

    elif ds_method == "oversample_block_average":
        f = int(oversample_factor)
        if f < 1:
            raise ValueError("oversample_factor must be >= 1")

        fine_size_vox = tuple(int(n * f) for n in final_size_vox)
        fine_spacing_mm = tuple(float(s / f) for s in final_spacing_mm)

        ct_fine = sitk.Resample(
            ct_f,
            fine_size_vox,
            translation,
            sitk.sitkLinear,
            final_origin,
            fine_spacing_mm,
            final_direction,
            0.0,
            sitk.sitkFloat32,
        )

        arr_fine_zyx = sitk.GetArrayFromImage(ct_fine).astype(np.float32)

        nx_c, ny_c, nz_c = final_size_vox
        expected_shape = (nz_c * f, ny_c * f, nx_c * f)
        if arr_fine_zyx.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected fine grid shape {arr_fine_zyx.shape}, expected {expected_shape}"
            )

        arr_zyx = arr_fine_zyx.reshape(
            nz_c, f,
            ny_c, f,
            nx_c, f
        ).mean(axis=(1, 3, 5)).astype(np.float32)

        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

        ct_coarse = sitk.GetImageFromArray(arr_zyx)
        ct_coarse.SetOrigin(tuple(final_origin))
        ct_coarse.SetSpacing(tuple(final_spacing_mm))
        ct_coarse.SetDirection(tuple(final_direction))

        ds_method_used = "oversample_block_average"

    else:
        raise ValueError(
            f"Unknown ds_method='{ds_method}'. "
            "Use 'baseline', 'bspline', or 'oversample_block_average'."
        )

    # ============================================================
    # Build mesh
    # ============================================================
    mesh = xr.DataArray(
        arr_xyz,
        dims=["x", "y", "z"],
        attrs={
            "origin": list(round(v, 3) for v in ct_coarse.GetOrigin()),
            "spacing": list(round(v, 3) for v in ct_coarse.GetSpacing()),
            "direction": list(round(v, 3) for v in ct_coarse.GetDirection()),
            "shift_mm": list(round(v, 3) for v in shift_mm),
            "mesh_represents": "mean_ct",
            "downsampling_method": ds_method_used,
            "remove_barrel": bool(remove_barrel),
        },
    )

    if ds_method == "oversample_block_average":
        mesh.attrs["oversample_factor"] = int(oversample_factor)

    if remove_barrel:
        mesh.attrs["barrel_mask_params"] = dict(barrel_mask_params)
        mesh.attrs["barrel_removed_voxels"] = barrel_removed_voxels

    # ============================================================
    # Export fractions
    # ============================================================
    if export_fractions:
        if th_air is None or th_filler is None:
            raise ValueError("export_fractions=True requires th_air and th_filler")

        os.makedirs(out_dir, exist_ok=True)

        arr_hi = sitk.GetArrayFromImage(ct_f).astype(np.float32)  # (z,y,x)
        seg = np.zeros_like(arr_hi, dtype=np.uint8)
        seg[arr_hi <= th_air] = 0
        seg[(arr_hi > th_air) & (arr_hi <= th_filler)] = 1
        seg[arr_hi > th_filler] = 2

        seg_img = sitk.GetImageFromArray(seg)
        seg_img.CopyInformation(ct_f)

        m_air = sitk.Cast(seg_img == 0, sitk.sitkFloat32)
        m_fill = sitk.Cast(seg_img == 1, sitk.sitkFloat32)
        m_metal = sitk.Cast(seg_img == 2, sitk.sitkFloat32)

        def res_frac(m):
            return sitk.Resample(
                m,
                final_size_vox,
                translation,
                sitk.sitkLinear,
                final_origin,
                final_spacing_mm,
                final_direction,
                0.0,
                sitk.sitkFloat32,
            )

        f_air_zyx = sitk.GetArrayFromImage(res_frac(m_air)).astype(np.float32)
        f_fill_zyx = sitk.GetArrayFromImage(res_frac(m_fill)).astype(np.float32)
        f_metal_zyx = sitk.GetArrayFromImage(res_frac(m_metal)).astype(np.float32)

        s = f_air_zyx + f_fill_zyx + f_metal_zyx + 1e-6
        f_air_zyx /= s
        f_fill_zyx /= s
        f_metal_zyx /= s

        f_air_xyz = np.transpose(f_air_zyx, (2, 1, 0))
        f_fill_xyz = np.transpose(f_fill_zyx, (2, 1, 0))
        f_metal_xyz = np.transpose(f_metal_zyx, (2, 1, 0))

        base = os.path.splitext(os.path.basename(ct_filepath))[0]
        suffix = "_nobarrel" if remove_barrel else ""

        p_air = os.path.join(out_dir, f"{base}{suffix}_f_air.npy")
        p_fill = os.path.join(out_dir, f"{base}{suffix}_f_fill.npy")
        p_metal = os.path.join(out_dir, f"{base}{suffix}_f_metal.npy")

        np.save(p_air, f_air_xyz)
        np.save(p_fill, f_fill_xyz)
        np.save(p_metal, f_metal_xyz)

        mesh.attrs["th_air"] = float(th_air)
        mesh.attrs["th_filler"] = float(th_filler)
        mesh.attrs["fraction_paths"] = {"f_air": p_air, "f_fill": p_fill, "f_metal": p_metal}
        mesh.attrs["fraction_note"] = "fractions from hi-res threshold -> coarse linear resample"

    return mesh