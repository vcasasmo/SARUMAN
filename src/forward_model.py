from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# forward_model.py
SYNTHETIC_CASE = False
BYPASS = False

import os
import numpy as np
import polars as pl
from tqdm import tqdm

from src.tools.io_tools import generate_uuid
from src.surrogates.detector_sumo import estimate_detector_efficiencies
from src.surrogates.ct_sumo import load_ct2lac_model
from src.prepare_debug import export_prepare_volume, should_export_volume

# ---- CT2LAC API (used only if BYPASS=False) ----

# For FAST PATH you must have added these in CT2LAC.py / ct_sumo.py:
#   api["get_Z_rho_vectorized"]  -> returns (Z_pred, rho_est) for ct_vals
#   api["mac_interp_by_Z"]       -> dict Z -> interp1d(E)->Mac
#   api["predict_Z_from_rho_and_ct"] -> recompute Z after rho correction
# If not present, this file will raise a KeyError.


def get_det_efficiency(waste_batch):
    # [pos][x][y][z][E][2]
    waste_batch.detector_efficiencies = np.zeros(
        (
            len(waste_batch.detector_positions),
            waste_batch.mesh.shape[0],
            waste_batch.mesh.shape[1],
            waste_batch.mesh.shape[2],
            len(waste_batch.energies_detected),
            2,
        ),
        dtype="float32",
    )
    estimate_detector_efficiencies(waste_batch)
    # Optional diagnostic exports. These do not modify the forward model.
    if should_export_volume(waste_batch, 0, 0, "ct"):
        export_prepare_volume(waste_batch, "ct_coarsened", waste_batch.mesh.values)
    for pos in range(waste_batch.detector_efficiencies.shape[0]):
        for energy_idx in range(waste_batch.detector_efficiencies.shape[4]):
            if should_export_volume(waste_batch, pos, energy_idx, "efficiency"):
                export_prepare_volume(
                    waste_batch, "efficiency",
                    waste_batch.detector_efficiencies[pos, :, :, :, energy_idx, 0],
                    pos, energy_idx,
                )


def adu2lac(waste_batch):
    import SimpleITK as sitk
    from scipy.interpolate import interp1d

    # Legacy BYPASS fractions are not needed for CT2LAC.
    # They are deliberately not loaded in the production branch.

    # -------------------------
    # Init transmission array (stores tau first)
    # -------------------------
    shape = (
        len(waste_batch.detector_positions),
        waste_batch.mesh.shape[0],
        waste_batch.mesh.shape[1],
        waste_batch.mesh.shape[2],
        len(waste_batch.energies_detected),
    )
    waste_batch.transmission = np.zeros(shape, dtype=np.float32)

    # -------------------------
    # Cache ray-paths per detector (avoid re-reading CSV per energy)
    # -------------------------
    paths_cache = []
    max_i, max_j, max_k = waste_batch.mesh.shape

    for detector_PoV in range(waste_batch.detector_positions.height):
        det_x, det_y, det_z = waste_batch.detector_positions.row(detector_PoV)
        unique_id = generate_uuid(det_x, det_y, det_z, "detector")

        detector_csv_path = os.path.join(waste_batch.mesh_folder_path, f"{unique_id}.csv")
        path = pl.read_csv(detector_csv_path, columns=["i", "j", "k", "d", "voxel_id"])

        i_values = path["i"].to_numpy()
        j_values = path["j"].to_numpy()
        k_values = path["k"].to_numpy()
        d_values = path["d"].to_numpy().astype(np.float32)  # mm
        voxel_ids_all = path["voxel_id"].to_numpy()

        if (i_values >= max_i).any() or (j_values >= max_j).any() or (k_values >= max_k).any():
            raise IndexError(
                f"Out of bounds indices: i_max={max_i}, j_max={max_j}, k_max={max_k}, Detector Pos. {detector_PoV}"
            )

        # NOTE:
        # voxel_id identifies the ORIGIN voxel (one ray per origin voxel), not the traversed voxel.
        # first_indices gives (for each voxel_id) a representative sample; in practice this corresponds
        # to the origin voxel coordinates to write the summed tau back.
        _, first_indices = np.unique(voxel_ids_all, return_index=True)
        voxel_ids = voxel_ids_all[first_indices]

        paths_cache.append((i_values, j_values, k_values, d_values, voxel_ids_all, first_indices, voxel_ids))

    # -------------------------
    # BYPASS branch: fractions -> LAC(E) -> tau
    # -------------------------
    if BYPASS or SYNTHETIC_CASE:
        fp = waste_batch.mesh.attrs["fraction_paths"]
        f_air = np.load(fp["f_air"]).astype(np.float32)
        f_fill = np.load(fp["f_fill"]).astype(np.float32)
        f_metal = np.load(fp["f_metal"]).astype(np.float32)
        filler_name = "Bark"
        filler_density = 0.25

        filler_mac_df = pl.read_csv((PROJECT_ROOT / f"src/data/mac/{filler_name}.csv"))
        air_mac_df = pl.read_csv((PROJECT_ROOT / "src/data/mac/Air.csv"))
        iron_mac_df = pl.read_csv((PROJECT_ROOT / "src/data/mac/Iron.csv"))

        i_fill = interp1d(filler_mac_df["E"], filler_mac_df["Mac"], kind="linear", fill_value="extrapolate")
        i_air = interp1d(air_mac_df["E"], air_mac_df["Mac"], kind="linear", fill_value="extrapolate")
        i_iron = interp1d(iron_mac_df["E"], iron_mac_df["Mac"], kind="linear", fill_value="extrapolate")

        rho_air, rho_fill, rho_iron = 0.001205, float(filler_density), 7.874

        for e_idx, energy_keV in enumerate(waste_batch.energies_detected):
            lac_air = float(i_air(energy_keV)) * rho_air
            lac_fill = float(i_fill(energy_keV)) * rho_fill
            lac_iron = float(i_iron(energy_keV)) * rho_iron

            for detector_PoV in tqdm(
                range(waste_batch.detector_positions.height),
                desc=f"Estimating transmission values (BYPASS, E={energy_keV:.2f} keV)",
            ):
                i_values, j_values, k_values, d_values, voxel_ids_all, first_indices, voxel_ids = paths_cache[
                    detector_PoV
                ]

                if SYNTHETIC_CASE:
                    LAC_values = waste_batch.mesh.values[i_values, j_values, k_values].astype(np.float32)
                else:
                    fa = f_air[i_values, j_values, k_values]
                    ff = f_fill[i_values, j_values, k_values]
                    fm = f_metal[i_values, j_values, k_values]
                    LAC_values = fa * lac_air + ff * lac_fill + fm * lac_iron

                tau_per_sample = (d_values / 10.0) * LAC_values  # mm->cm

                max_voxel_id = voxel_ids_all.max()
                order_sums = np.bincount(voxel_ids_all, weights=tau_per_sample, minlength=max_voxel_id + 1)

                # write tau to the ORIGIN voxel coords for each voxel_id (from representative samples)
                i_vals = i_values[first_indices]
                j_vals = j_values[first_indices]
                k_vals = k_values[first_indices]
                waste_batch.transmission[detector_PoV, i_vals, j_vals, k_vals, e_idx] = order_sums[voxel_ids]

                if should_export_volume(waste_batch, detector_PoV, e_idx, "attenuation"):
                    export_prepare_volume(waste_batch, "attenuation_tau", waste_batch.transmission[detector_PoV, :, :, :, e_idx], detector_PoV, e_idx)

    # -------------------------
    # CT2LAC branch: FAST PATH (correct handling of voxel_id)
    # -------------------------
    else:
        api = load_ct2lac_model(config=getattr(waste_batch, "config", None))
        # Precompute rho and Z for entire mesh ONCE (independent of detector and energy)
        ct_grid = waste_batch.mesh.values.astype(np.float32)  # (x,y,z)
        ct_flat = ct_grid.ravel()

        Z_flat, rho_flat = api["get_Z_rho_vectorized"](ct_flat)
        Z_grid = Z_flat.reshape(ct_grid.shape)
        rho_grid = rho_flat.reshape(ct_grid.shape).astype(np.float32)

        # -------------------------
        # Mass diagnostic / optional density calibration.
        # No declared drum mass is assumed for an arbitrary waste package.
        # The default (enabled=False) preserves the previously validated rho_grid.
        # -------------------------
        mass_cfg = (getattr(waste_batch, "config", None) or {}).get("mass_calibration", {})
        apply_mass_scaling = bool(mass_cfg.get("enabled", False))
        declared_gross_mass_kg = mass_cfg.get("declared_gross_mass_kg")
        drum_mass_kg = float(mass_cfg.get("empty_drum_mass_kg", 19.0))
        spacing_mm = waste_batch.mesh.attrs.get("spacing", (19.2, 19.2, 19.2))
        voxel_volume_cm3 = float(np.prod(spacing_mm)) / 1000.0
        content_mask = rho_grid > 0.0
        mass_before_kg = float(np.sum(rho_grid[content_mask]) * voxel_volume_cm3 / 1000.0)
        print(f"[MASS] CT-estimated content mass = {mass_before_kg:.2f} kg")

        if apply_mass_scaling and declared_gross_mass_kg is None:
            raise ValueError("mass_calibration.enabled=true requires declared_gross_mass_kg")
        if declared_gross_mass_kg is not None:
            target_content_mass_kg = float(declared_gross_mass_kg) - drum_mass_kg
            if target_content_mass_kg <= 0 or mass_before_kg <= 0:
                raise ValueError("Mass calibration requires positive target and CT-estimated mass")
            alpha = target_content_mass_kg / mass_before_kg
            print(f"[MASS] Declared gross mass = {float(declared_gross_mass_kg):.2f} kg")
            print(f"[MASS] Empty drum mass     = {drum_mass_kg:.2f} kg")
            print(f"[MASS] Target content mass = {target_content_mass_kg:.2f} kg")
            print(f"[MASS] Diagnostic ratio    = {alpha:.4f}")
            if apply_mass_scaling:
                rho_grid[content_mask] *= alpha
                print("[MASS] Density scaling explicitly enabled")
            else:
                print("[MASS] Density scaling disabled; CT density unchanged")
        else:
            print("[MASS] No declared mass supplied; density scaling disabled")

        # Keep the original Z re-evaluation step for numerical compatibility.
        Z_flat_corr = api["predict_Z_from_rho_and_ct"](
            rho_grid.ravel(),
            ct_grid.ravel(),
        )
        Z_grid = Z_flat_corr.reshape(ct_grid.shape)
        mass_after_kg = float(np.sum(rho_grid[content_mask]) * voxel_volume_cm3 / 1000.0)
        print(f"[MASS] Content mass after  = {mass_after_kg:.2f} kg")

        mac_interp_by_Z = api["mac_interp_by_Z"]  # dict: Z(int) -> interp1d(E)->Mac

        nx, ny, nz = waste_batch.mesh.shape
        nxny = nx * ny

        for e_idx, energy_keV in enumerate(waste_batch.energies_detected):
            energy_arr = np.array([energy_keV], dtype=np.float32)

            for detector_PoV in tqdm(
                range(waste_batch.detector_positions.height),
                desc=f"Estimating transmission values (CT2LAC FAST, E={energy_keV:.2f} keV)",
            ):
                i_values, j_values, k_values, d_values, voxel_ids_all, first_indices, voxel_ids = paths_cache[
                    detector_PoV
                ]

                # --- UNIQUE TRAVERSED VOXELS (keyed by (i,j,k), NOT voxel_id) ---
                lin = i_values + nx * j_values + nxny * k_values
                lin_u, inv_lin = np.unique(lin, return_inverse=True)

                i_u = lin_u % nx
                j_u = (lin_u // nx) % ny
                k_u = lin_u // nxny

                rho_u = rho_grid[i_u, j_u, k_u]
                Z_u = Z_grid[i_u, j_u, k_u]

                # LAC per unique traversed voxel: rho * MAC(Z,E)
                lac_u = np.empty_like(rho_u, dtype=np.float32)
                for Z in np.unique(Z_u):
                    mac = float(mac_interp_by_Z[int(Z)](energy_arr)[0])
                    mZ = (Z_u == Z)
                    lac_u[mZ] = rho_u[mZ] * mac
                lac_u = np.clip(lac_u, 0.0, None).astype(np.float32)

                # expand to per-sample LAC
                LAC_values = lac_u[inv_lin]
                tau_per_sample = (d_values / 10.0) * LAC_values  # mm->cm

                # sum tau per ray/origin (voxel_id)
                max_voxel_id = voxel_ids_all.max()
                order_sums = np.bincount(voxel_ids_all, weights=tau_per_sample, minlength=max_voxel_id + 1)

                # write tau to the ORIGIN voxel coords for each voxel_id (from representative samples)
                i_vals = i_values[first_indices]
                j_vals = j_values[first_indices]
                k_vals = k_values[first_indices]
                waste_batch.transmission[detector_PoV, i_vals, j_vals, k_vals, e_idx] = order_sums[voxel_ids]

                if should_export_volume(waste_batch, detector_PoV, e_idx, "attenuation"):
                    export_prepare_volume(waste_batch, "attenuation_tau", waste_batch.transmission[detector_PoV, :, :, :, e_idx], detector_PoV, e_idx)

    # Convert tau -> transmission
    waste_batch.transmission = np.exp(-waste_batch.transmission).astype(np.float32)
