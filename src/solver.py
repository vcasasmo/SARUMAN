from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #to allow to run tests from here using Saruman

def solver(isotope, trial, num_iterations, mockup_case, omit_zero_observations=True, exclude_eu154_715=True,
           *, input_path=None, output_dir=None, config=None, detector_csv=None):
    import os
    import numpyro
    import numpyro.distributions as dist
    import numpy as np
    import matplotlib.pyplot as plt

    
    
    from src.prepare_cli import Saruman
    from pathlib import Path
    from jax import random
    #jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    import polars as pl


    import seaborn as sns 

    print("Running SARUMAN Bayesian SVI reconstruction")
    # Preserve the established numerical model; only expose its existing defaults.
    solver_cfg = (config or {}).get("solver", {})
    prior_cfg = solver_cfg.get("prior", {})
    optimization_cfg = solver_cfg.get("optimization", {})
    n_samples = int(solver_cfg.get("posterior_samples", 1000))
    seed = int(solver_cfg.get("seed", 20202))
    if n_samples < 1:
        raise ValueError("posterior_samples must be positive")

    # Allow both single-isotope and joint multi-isotope calls.
    # Examples:
    #   solver("Cs-137", trial, num_iterations, "6.h5")
    #   solver(["Eu-152", "Eu-154", "Ag-110M"], trial, num_iterations, "1.h5")
    if isinstance(isotope, (list, tuple, set)):
        isotope_list = [str(x) for x in isotope]
    else:
        isotope_list = [str(isotope)]
    isotope_label = "__".join([x.replace("/", "-") for x in isotope_list])
    is_joint = len(isotope_list) > 1

    
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import Adam
    """Environment variables for device setting"""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    
    def init_solver(waste_batch, isotope_list, exclude_eu154_715=True):
     
    
        # ----------------------------
        # Helper: length in hollow cylinder wall (JAX)
        # ----------------------------
        def hollow_cylinder_wall_length_jnp(p0, p1, Ro, Ri, center_xy=(0.0, 0.0), eps=1e-12):
            cx, cy = center_xy
            p0 = p0.at[..., 0].add(-cx)
            p0 = p0.at[..., 1].add(-cy)
            p1 = p1.at[..., 0].add(-cx)
            p1 = p1.at[..., 1].add(-cy)
    
            d = p1 - p0
            seg_len = jnp.linalg.norm(d, axis=-1) + eps
            v = d / seg_len[..., None]
    
            x0, y0 = p0[..., 0], p0[..., 1]
            vx, vy = v[..., 0], v[..., 1]
    
            a = vx * vx + vy * vy
            r0_sq = x0 * x0 + y0 * y0
            r0 = jnp.sqrt(r0_sq + eps)
    
            def interval_inside_radius(R):
                c = r0_sq - R * R
                b = 2.0 * (x0 * vx + y0 * vy)
                disc = b * b - 4.0 * a * c
    
                a_small = a < eps
                inside_const = (r0 <= R)
                lo_const = jnp.zeros_like(seg_len)
                hi_const = seg_len
    
                disc_pos = disc > 0.0
                sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
                s1 = (-b - sqrt_disc) / (2.0 * a + eps)
                s2 = (-b + sqrt_disc) / (2.0 * a + eps)
                lo_line = jnp.minimum(s1, s2)
                hi_line = jnp.maximum(s1, s2)
    
                lo = jnp.maximum(lo_line, 0.0)
                hi = jnp.minimum(hi_line, seg_len)
                valid_seg = (hi > lo)
    
                inside_all = (r0 <= R)
    
                lo_out = jnp.where(a_small, lo_const, jnp.where(disc_pos, lo, lo_const))
                hi_out = jnp.where(a_small, hi_const, jnp.where(disc_pos, hi, hi_const))
    
                valid_out = jnp.where(
                    a_small,
                    inside_const,
                    jnp.where(disc_pos, valid_seg, inside_all)
                )
                return lo_out, hi_out, valid_out
    
            loO, hiO, vO = interval_inside_radius(Ro)
            loI, hiI, vI = interval_inside_radius(Ri)
    
            lenO = jnp.where(vO, jnp.maximum(0.0, hiO - loO), 0.0)
    
            left = jnp.maximum(0.0, jnp.minimum(loI, hiO) - loO)
            left = jnp.where(loI > loO, left, 0.0)
    
            right = jnp.maximum(0.0, hiO - jnp.maximum(hiI, loO))
            right = jnp.where(hiI < hiO, right, 0.0)
    
            wall_when_both = left + right
            wall = jnp.where(vO & vI, wall_when_both, jnp.where(vO & (~vI), lenO, 0.0))
            return wall
    
        # ----------------------------
        # 0) Shapes: transmission = (pos, x, y, z, E)
        # ----------------------------
        tsh = waste_batch.transmission.shape
        if len(tsh) != 5:
            raise ValueError(f"Expected transmission 5D (pos,x,y,z,E). Got {tsh}")
    
        Npos, Nx, Ny, Nz, NE_all = tsh
    
        # ----------------------------
        # 1) Indices / energies for this isotope
        # ----------------------------
        isotopes = list(waste_batch.isotopes_detected)
        isotope_set = set(isotope_list)
        all_energies = np.asarray(waste_batch.energies_detected, dtype=float)
        candidate_energy_indices = [i for i, element in enumerate(isotopes) if element in isotope_set]

        # Sensitivity-cleaning rule: Eu-154 at 715.76001 keV is excluded as a
        # problematic gamma line, rather than changing its nuclear emission
        # probability. This removes the complete line from likelihood and PPC.
        eu154_problematic_energy_keV = 715.76001
        eu154_problematic_tolerance_keV = 0.25
        excluded_gamma_line_rows = []
        energy_indices = []
        for idx in candidate_energy_indices:
            iso_name = isotopes[idx]
            energy_keV = float(all_energies[idx])
            is_problematic_eu154 = (
                exclude_eu154_715
                and iso_name == "Eu-154"
                and abs(energy_keV - eu154_problematic_energy_keV) <= eu154_problematic_tolerance_keV
            )
            if is_problematic_eu154:
                excluded_gamma_line_rows.append({
                    "ags_energy_keV": energy_keV,
                    "isotope": iso_name,
                    "omission_reason": "excluded_problematic_Eu154_715.76001_keV",
                })
            else:
                energy_indices.append(idx)

        energies = all_energies[energy_indices]
        selected_isotopes = [isotopes[i] for i in energy_indices]
        isotope_to_idx = {iso: k for k, iso in enumerate(isotope_list)}
        peak_to_iso = np.array([isotope_to_idx[iso] for iso in selected_isotopes], dtype=np.int32)
        nE = len(energy_indices)
    
        # ----------------------------
        # 2) Detector efficiency: queremos (pos,x,y,z,nE)
        #    Tu array puede venir en varios órdenes; probamos el más lógico primero.
        # ----------------------------
        eff = waste_batch.detector_efficiencies
        eff_sh = eff.shape
    
        # Caso A (ideal): (pos,x,y,z,E,1)
        if len(eff_sh) == 6 and eff_sh[:5] == (Npos, Nx, Ny, Nz, NE_all):
            det_efficiency = eff[:, :, :, :, energy_indices, 0]
    
        # Caso B: (x,y,z,pos,E,1)  (lo que tenías antes pero con pos)
        elif len(eff_sh) == 6 and eff_sh[:5] == (Nx, Ny, Nz, Npos, NE_all):
            det_efficiency = eff[:, :, :, :, energy_indices, 0]           # (x,y,z,pos,nE)
            det_efficiency = jnp.transpose(det_efficiency, (3, 0, 1, 2, 4))  # -> (pos,x,y,z,nE)
    
        else:
            raise ValueError(
                f"detector_efficiencies shape {eff_sh} no encaja con transmission {tsh}.\n"
                f"Espero (pos,x,y,z,E,1) o (x,y,z,pos,E,1)."
            )
    
        det_efficiency = jnp.clip(det_efficiency, a_min=0)  # (pos,x,y,z,nE)
    
        # ----------------------------
        # 3) Drum wall geometry (mm)
        # ----------------------------
        Ro = 287.0   # mm
        t_wall = 1.0 # mm
        Ri = Ro - t_wall
        center_xy = (0.0, 0.0)
        # Legacy large debug export removed from production runs
        # --- voxel centers from mesh coords (mm) ---
        x0 = jnp.array(waste_batch.mesh.coords["x"].values)
        y0 = jnp.array(waste_batch.mesh.coords["y"].values)
        z0 = jnp.array(waste_batch.mesh.coords["z"].values)
    
        sx, sy, sz =19.2,19.2,19.2 # mm
    
        # sanity sizes
        if x0.shape[0] != Nx or y0.shape[0] != Ny or z0.shape[0] != Nz:
            raise ValueError(
                f"mesh coords sizes x,y,z = {x0.shape[0], y0.shape[0], z0.shape[0]} "
                f"do not match transmission x,y,z = {(Nx,Ny,Nz)}"
            )
    
        x = x0 + 0.5 * float(sx)
        y = y0 + 0.5 * float(sy)
        z = z0 + 0.5 * float(sz)
    
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        vox_pos = jnp.stack([X, Y, Z], axis=-1)      # (x,y,z,3)
    
        # Broadcast:
        # vox_pos_b: (1, x, y, z, 3)
        # det_pos_b: (pos, 1, 1, 1, 3)
        vox_pos_b = vox_pos[None, :, :, :, :]  # (1,Nx,Ny,Nz,3)
    
        # --- detector positions (mm) from CSV ---
        det_df = pl.read_csv(str(detector_csv or Path(__file__).resolve().parent / "data/detector_coordinates.csv"))
        det_np = det_df.select(["x", "y", "z"]).to_numpy()
        if det_np.shape[0] != Npos:
            raise ValueError(
                f"detector_coordinates.csv has {det_np.shape[0]} rows but transmission expects Npos={Npos}."
            )
        det_pos = jnp.array(det_np)  # (pos,3)
        det_pos_b = det_pos[:, None, None, None, :]  # (pos,1,1,1,3)
    
        # --- drum steel attenuation from iron MAC table ---
        # The CT/transmission already accounts for the waste matrix. T_extra only accounts
        # for the extra attenuation in the drum steel wall, energy by energy, using iron as
        # a steel approximation.
        iron_mac_path = str(Path(__file__).resolve().parent / "data/mac/Iron.csv")
        iron_density_g_cm3 = 7.874  # iron density; approximation for carbon steel

        def load_iron_mu_1_per_cm(target_energies_keV):
            """Return linear attenuation coefficients for iron at target energies.

            The MAC file is expected to contain an energy column and a mass attenuation
            coefficient column. The reader is intentionally tolerant to column names.
            Energies are converted from MeV to keV if the table looks MeV-based.
            """
            if not os.path.exists(iron_mac_path):
                raise FileNotFoundError(
                    f"Iron MAC file not found: {iron_mac_path}. "
                    "Required to compute T_extra for real drum steel attenuation."
                )

            mac_df = pl.read_csv(iron_mac_path)
            if mac_df.width < 2:
                raise ValueError(f"Iron MAC table must contain at least two columns: {iron_mac_path}")

            cols = mac_df.columns
            lower = {c: c.lower().replace(' ', '').replace('-', '').replace('_', '') for c in cols}
            energy_col = next((c for c in cols if 'energy' in lower[c] or lower[c] in {'e', 'emev', 'ekev'}), cols[0])
            mac_col = next((
                c for c in cols
                if c != energy_col and (
                    'mac' in lower[c] or 'massattenuation' in lower[c]
                    or 'muoverrho' in lower[c] or 'mucm2g' in lower[c]
                )
            ), cols[1])

            table = (
                mac_df
                .select([pl.col(energy_col).cast(pl.Float64).alias('energy'),
                         pl.col(mac_col).cast(pl.Float64).alias('mac_cm2_g')])
                .drop_nulls()
                .sort('energy')
            )
            e_table = table['energy'].to_numpy()
            mac_table = table['mac_cm2_g'].to_numpy()
            if len(e_table) == 0:
                raise ValueError(f"Iron MAC table has no valid numeric rows: {iron_mac_path}")

            target_energies_keV = np.asarray(target_energies_keV, dtype=float)
            if np.nanmax(e_table) <= 50.0 and np.nanmax(target_energies_keV) > 50.0:
                e_table = e_table * 1000.0

            mac_interp = np.interp(target_energies_keV, e_table, mac_table,
                                   left=mac_table[0], right=mac_table[-1])
            return mac_interp * iron_density_g_cm3

        if nE == 0:
            raise ValueError(f"No retained detected energies found for isotope vector {isotope_list!r}.")

        mu_steel_cm = jnp.array(load_iron_mu_1_per_cm(energies), dtype=jnp.float32)  # (nE,)
        mu_steel_mm = mu_steel_cm / 10.0  # 1/cm -> 1/mm
        # Debug steel attenuation export disabled for production

        # --- compute extra transmission due to drum wall ---
        L_steel_mm = hollow_cylinder_wall_length_jnp(
            vox_pos_b, det_pos_b, Ro=Ro, Ri=Ri, center_xy=center_xy
        )  # (pos,x,y,z)
    
        T_extra = jnp.exp(-L_steel_mm[..., None] * mu_steel_mm[None, None, None, None, :])
        # (pos,x,y,z,nE)
    
        # ----------------------------
        # 4) Transmission + w
        # ----------------------------
        transmission = waste_batch.transmission[:, :, :, :, energy_indices]* T_extra  # (pos,x,y,z,nE)
        # Debug transmission export disabled for production
        w = transmission * det_efficiency                                               # (pos,x,y,z,nE)
        # Debug efficiency export disabled for production
        # Debug response export disabled for production
        # ----------------------------
        # 5) Measurement times
        # ----------------------------
        t = waste_batch.spc_time.flatten()
    
        # ----------------------------
        # 6) Load emission probabilities eta, preserving AGS energy order.
        # ----------------------------
        # Important for joint inference: eta[i] must correspond exactly to
        # energies[i], selected_isotopes[i], w[..., i] and obs[i, :].
        tolerance = 0.15
        gammaspec_cache = {}

        def gammaspec_path_for_iso(iso_name):
            candidates = [
                f"{Path(__file__).resolve().parent / 'data/gammaspec'}/{iso_name.replace('-', '')}.csv",
                f"{Path(__file__).resolve().parent / 'data/gammaspec'}/{iso_name.replace('-', '').replace('m', 'M')}.csv",
                f"{Path(__file__).resolve().parent / 'data/gammaspec'}/{iso_name.replace('-', '').replace('M', 'm')}.csv",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            raise FileNotFoundError(f"No gamma spec CSV found for {iso_name}. Tried: {candidates}")

        def get_gammaspec(iso_name):
            if iso_name not in gammaspec_cache:
                path = gammaspec_path_for_iso(iso_name)
                gammaspec_cache[iso_name] = pl.read_csv(path, columns=["energy", "intensity_%"]).sort("energy")
            return gammaspec_cache[iso_name]

        eta_values = []
        eta_audit_rows = []
        for E, iso_name in zip(energies, selected_isotopes):
            gd = get_gammaspec(iso_name)
            match = (
                gd.with_columns((pl.col("energy") - float(E)).abs().alias("delta_keV"))
                  .filter(pl.col("delta_keV") <= tolerance)
                  .sort("delta_keV")
            )
            if match.height == 0:
                nearest = gd.with_columns((pl.col("energy") - float(E)).abs().alias("delta_keV")).sort("delta_keV").head(3)
                raise ValueError(f"No gamma-spec match for {iso_name} at {E:.3f} keV within {tolerance} keV. Nearest:\n{nearest}")
            row = match.row(0, named=True)
            eta_val = float(row["intensity_%"]) / 100.0
            eta_values.append(eta_val)
            eta_audit_rows.append({
                "ags_energy_keV": float(E),
                "isotope": iso_name,
                "matched_energy_keV": float(row["energy"]),
                "delta_keV": float(row["delta_keV"]),
                "eta": float(eta_val),
                "status": "OK" if match.height == 1 else f"MULTIPLE_MATCHES_{match.height}",
            })

        eta = np.array(eta_values, dtype=float)
        eta_audit = pl.DataFrame(eta_audit_rows)
        print("Gamma eta audit:")
        print(eta_audit)
    
        # ----------------------------
        # 7) Observations and per-observation traceability
        # ----------------------------
        obs_matrix = np.asarray(waste_batch.spectrum_analyzed[energy_indices, :, 0], dtype=float)
        obs_sigma_matrix = np.asarray(waste_batch.spectrum_analyzed[energy_indices, :, 1], dtype=float)
        n_positions = obs_matrix.shape[1]
        observation_metadata = {
            "isotope": np.repeat(np.asarray(selected_isotopes, dtype=object), n_positions),
            "energy_keV": np.repeat(np.asarray(energies, dtype=float), n_positions),
            "detector_index_within_peak": np.tile(np.arange(n_positions, dtype=int), nE),
        }
        excluded_gamma_lines = pl.DataFrame(
            excluded_gamma_line_rows,
            schema={
                "ags_energy_keV": pl.Float64,
                "isotope": pl.Utf8,
                "omission_reason": pl.Utf8,
            },
        )
    
        # ----------------------------
        # 8) numpy2jax
        # ----------------------------
        obs = jax.device_put(jnp.array(obs_matrix.flatten()))
        obs_sigma = jax.device_put(jnp.array(obs_sigma_matrix.flatten()))
        w = jax.device_put(jnp.array(w))
        eta = jax.device_put(jnp.array(eta))
        t = jax.device_put(jnp.array(t))
        return (
            t, eta, w, obs, obs_sigma, jnp.array(peak_to_iso), len(isotope_list),
            selected_isotopes, energies, eta_audit, observation_metadata, excluded_gamma_lines
        )




 
    
    def mask_generation(waste_batch):
        """Generates a mask for voxels inside or touching a cylinder of radius R and height H centered at the origin.
        Now it is hard coded but in the future can be modifyied to allow for flexibility"""
        #TODO: chane R H or even shape according to Context Manager, same for XRprocessor
        # Cylinder dimensions
        R = 287  # Cylinder radius
        H = 885  # Cylinder height
        mesh=waste_batch.mesh
        # Extract mesh coordinates and spacing
        x = mesh['x'].values
        y = mesh['y'].values
        z = mesh['z'].values
    
        # Create a grid of voxel centers (xw, y, z)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
        # Condition for the cylinder:
        # - The radial distance sqrt(x^2 + z^2) must be less than or equal to R.
        # - The y-coordinate (vertical) must lie within [-H/2, H/2] (since the cylinder is centered at the origin).
        condition = (np.sqrt(xx**2 + zz**2) <= R) & (np.abs(yy) <= H / 2)
        mask = np.where(condition, 1, 0).astype(np.uint8)
    
        return mask
     
    
    def ppc_by_index(y_pred, obs, n_draws=30, title="PPC by detector index"):
        import numpy as np
        import jax.numpy as jnp
        import matplotlib.pyplot as plt
    
        # y_pred: (S, N)
        S, N = y_pred.shape
        idx = jnp.arange(N)
    
        mean = jnp.mean(y_pred, axis=0)
        median = jnp.median(y_pred, axis=0)
        lo = jnp.percentile(y_pred, 2.5, axis=0)
        hi = jnp.percentile(y_pred, 97.5, axis=0)
    
        plt.figure(figsize=(12, 5))
        plt.plot(obs, color="black", linewidth=2, label="Observed")
    
        # random draws
        draws = np.random.choice(S, size=min(n_draws, S), replace=False)
        for s in draws:
            plt.plot(y_pred[s], alpha=0.15, linewidth=1)
    
        plt.fill_between(np.array(idx), np.array(lo), np.array(hi), alpha=0.25, label="95% PI")
        plt.plot(mean, linestyle="--", linewidth=2, label="Posterior mean")
        plt.plot(median, linestyle=":", linewidth=2, label="Posterior median")
    
        plt.xlabel("Detector index")
        plt.ylabel("Counts")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()     
    
     
    def posterior_predictive_check(
        y_pred, obs, obs_sigma, used_in_likelihood, observation_metadata, results_path, Save='Yes'
    ):
        """Posterior predictive check restricted to observations used in likelihood.

        Zero-valued non-reported peaks and excluded gamma lines are not part of the
        main PPC. Zero-valued omissions are documented separately in
        omitted_observations_audit.csv; gamma-line exclusions in
        excluded_gamma_lines.csv.
        """
        y_pred_np = np.asarray(y_pred)
        obs_np = np.asarray(obs)
        obs_sigma_np = np.asarray(obs_sigma)
        used = np.asarray(used_in_likelihood, dtype=bool)

        if used.shape[0] != obs_np.shape[0]:
            raise ValueError(f"Likelihood mask length {used.shape[0]} does not match observations {obs_np.shape[0]}.")
        if not np.any(used):
            raise ValueError("No observations remain in PPC after applying likelihood mask.")

        obs_used = obs_np[used]
        obs_sigma_used = obs_sigma_np[used]
        pred_used = y_pred_np[:, used]
        pred_mean = np.mean(pred_used, axis=0)
        pred_lower = np.percentile(pred_used, 2.5, axis=0)
        pred_upper = np.percentile(pred_used, 97.5, axis=0)

        order = np.argsort(obs_used)
        obs_sorted = obs_used[order]
        sigma_sorted = obs_sigma_used[order]
        mean_sorted = pred_mean[order]
        lower_sorted = pred_lower[order]
        upper_sorted = pred_upper[order]

        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
        ax.vlines(obs_sorted, lower_sorted, upper_sorted, color='blue', alpha=0.20, linewidth=1.0, label='95% PI')
        ax.errorbar(
            obs_sorted, mean_sorted, xerr=sigma_sorted, fmt='o', ms=3.5,
            color='blue', ecolor='0.55', elinewidth=0.7, capsize=1.5,
            alpha=0.80, label=r'Posterior mean ± observed $\sigma$'
        )
        min_val = float(min(np.min(obs_sorted), np.min(lower_sorted), np.min(mean_sorted)))
        max_val = float(max(np.max(obs_sorted), np.max(upper_sorted), np.max(mean_sorted)))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2, label='Perfect prediction')
        ax.set_xlabel("Observed Counts")
        ax.set_ylabel("Posterior Predicted Counts")
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.legend(fontsize=8)
        fig.tight_layout()

        if Save == 'Yes':
            used_metadata = {key: np.asarray(value)[used] for key, value in observation_metadata.items()}
            datasave = pl.DataFrame({
                'isotope': used_metadata['isotope'].tolist(),
                'energy_keV': used_metadata['energy_keV'].tolist(),
                'detector_index_within_peak': used_metadata['detector_index_within_peak'].tolist(),
                'obs': obs_used.tolist(),
                'obs_sigma': obs_sigma_used.tolist(),
                'y_pred_mean': pred_mean.tolist(),
                'y_pred_lower': pred_lower.tolist(),
                'y_pred_upper': pred_upper.tolist(),
            })
            datasave.write_csv(os.path.join(results_path, 'post_pred_check.csv'))
            fig.savefig(os.path.join(results_path, 'post_pred_check_plot.png'), format='png', dpi=600, bbox_inches='tight')

        plt.show()
        plt.close(fig)

    def total_recovered_activity(A_samples, prior_samples, save_path=None):
        """
        Plot the posterior distribution of total recovered activity.

        Posterior samples of total activity, with no external reference.
        """
        activity_sum = jnp.sum(A_samples, axis=(1, 2, 3)) / 1e3  # kBq

        plt.style.use("bmh")
        plt.figure(figsize=(8, 6))
        sns.histplot(activity_sum, bins=80, kde=True, stat="density", alpha=0.65, color="blue")

        plt.xlabel("Retrieved Activity (kBq)")
        plt.ylabel("Density")
        plt.grid(False)

        if save_path is not None:
            plt.savefig(save_path + '/total_activity.jpg', format='jpg', dpi=300)
            plt.show()
        else:
            plt.show()
        plt.close()
    
    
     
     
    
     
    def forward_model(w, eta, t, mask, peak_to_iso, n_isotopes, likelihood_mask=None, obs=None):
        #Beta Level:
        alpha = float(prior_cfg.get("beta_alpha", 0.1))
        beta = float(prior_cfg.get("beta_beta", 20.0))
        #Magnitude
        L = float(prior_cfg.get("magnitude_exponent_lower", 0.0))
        U = float(prior_cfg.get("magnitude_exponent_upper", 9.0))
        #Fine Level:
        std = 1.253314
        #Noise:
        param_1_gamma = float(prior_cfg.get("noise_gamma_shape", 1000.0))
        param_2_gamma = float(prior_cfg.get("noise_gamma_rate", 999.0))
        
        numpyro.deterministic("alpha", alpha)
        numpyro.deterministic("beta", beta)
        numpyro.deterministic("L", L)
        numpyro.deterministic("U", U)
        numpyro.deterministic("std_fine",std)
        numpyro.deterministic("param_1_gamma", param_1_gamma)
        numpyro.deterministic("param_2_gamma", param_2_gamma)

            
        #total inferred parameters 49152
        
        
        #number of energy peaks matches w 4th dimension:
        num_peaks = w.shape[4]            
        # Define original grid size
        I_orig, J_orig, K_orig = 31, 48, 31
      
        # Define downsampled grid size with padding
        I_padded = I_orig + (I_orig % 3)
        J_padded = J_orig + (J_orig % 3)
        K_padded = K_orig + (K_orig % 3)
        
        I_coarse = I_padded // 2
        J_coarse = J_padded // 2
        K_coarse = K_padded // 2
         
        #magnitude inference for the overall drum
        magnitude =10**numpyro.sample("magnitude",dist.Uniform(L,U))
        magnitude*= 1.538461384615383

        # Simple activity-vector layer above the shared multiresolution spatial field.
        # For joint inference this estimates total isotope proportions. For single-isotope
        # calls it is fixed to [1.0].
        if n_isotopes > 1:
            isotope_fractions = numpyro.sample(
                "isotope_fractions",
                dist.Dirichlet(jnp.ones((n_isotopes,)))
            )
        else:
            isotope_fractions = jnp.ones((1,))
        numpyro.deterministic("isotope_fractions_check", isotope_fractions)
    
        
    
        A_flat = numpyro.sample("A_coarse", dist.Beta(alpha, beta).expand([I_coarse, J_coarse, K_coarse])) 
        
        A_coarse = A_flat.reshape((I_coarse, J_coarse, K_coarse))
        
        
        # Expand to fine grid with hierarchical inference
        I_fine, J_fine, K_fine = I_coarse * 2, J_coarse * 2, K_coarse * 2
        
       
        # Broadcast coarse values to fine grid shape
        A_coarse_broadcasted = jnp.kron(A_coarse, jnp.ones((2, 2, 2)))
        #Normalize Beta level
        A_coarse_broadcasted /= jnp.sum(A_coarse_broadcasted)
        
        # Sample fine voxels conditioned on their coarse parent:
        with numpyro.plate("fine_voxels", I_fine * J_fine * K_fine):
            A_fine_flat = numpyro.sample(
                  "A_fine",  dist.HalfNormal(std)) *A_coarse_broadcasted.flatten()
            
       
       
        A_fine_full =A_fine_flat.reshape((I_fine, J_fine, K_fine))
        """
        Mask used for comparing ideal results vs. inferred by forcing inference only
        in voxels of the reference. For instance for Cs137_1_2.pkl:
        infer_mask = jnp.zeros((I_orig, J_orig, K_orig))
        infer_mask = infer_mask.at[23, 34, 23].set(1.0)
        """
        # Crop to original size
        infer_mask = jnp.zeros((I_orig, J_orig, K_orig))
        infer_mask = infer_mask.at[15, 24, 15].set(1.0) #central point
        #infer_mask = infer_mask.at[12, :, 15].set(1.0) # m11 
        #infer_mask = infer_mask.at[7, :, 10].set(1.0) #m14m15
       # infer_mask = infer_mask.at[20, :, 28].set(1.0) #m14m15
        A_fine = A_fine_full[:I_orig, :J_orig, :K_orig] * mask # *infer_mask
    
        numpyro.deterministic('A_field', A_fine)
        #Scaling for inferred magnitude
        A_fine*=magnitude 
        
        numpyro.deterministic('magnit', magnitude)
        
        # Save fine-level field
        numpyro.deterministic("A", A_fine)  # total activity field shared by all isotopes
        total_activity_kBq = jnp.sum(A_fine) / 1e3
        numpyro.deterministic("total_activity_kBq", total_activity_kBq)
        numpyro.deterministic("isotope_total_activity_kBq", total_activity_kBq * isotope_fractions)
        # Stack multiplicity for  handling multiple peaks
        A_expanded = jnp.stack([A_fine] * num_peaks, axis=-1)  
     
        # Compute forward model
        

        numpyro.deterministic('eff_correction_factors',1)
 
        peak_fractions = isotope_fractions[peak_to_iso]
        s_nm = t[:,None] * eta[None,:] * peak_fractions[None,:] * jnp.sum(w* A_expanded, axis=(1, 2, 3))
        s_nm = s_nm.T.flatten()     
        s_nm = s_nm+0.01
        # Noise to account for model error
        noise =numpyro.sample("noise",dist.Gamma(param_1_gamma,param_2_gamma))
        numpyro.deterministic('noise_check', noise)
        
        # Values omitted from the likelihood are still generated internally when
        # sampling, but are removed from the saved/evaluated PPC below.
        if likelihood_mask is None:
            likelihood_mask = jnp.ones_like(s_nm, dtype=bool)
        with numpyro.plate('data', s_nm.shape[0]):
            return numpyro.sample(
                "Y_obs",
                dist.TruncatedNormal(s_nm, jnp.sqrt(s_nm) * noise, low=0).mask(likelihood_mask),
                obs=obs
            )
    
    def prepare_results_folder(mockup_case,isotope, trial_number):
        import os
        if output_dir is not None:
            results_path = str(output_dir)
        else:
            folder_name = f"{os.path.basename(str(mockup_case))}_{isotope}_trial_{trial_number}"
            results_path = os.path.join(os.getcwd(), "results", folder_name)
        os.makedirs(results_path, exist_ok=True)
 
        return results_path
        
        
     
    
    #Problem variables    
    mockup_case=mockup_case
    isotope = isotope_label
    trial_number = trial
    results_path = prepare_results_folder(mockup_case, isotope,trial_number)
    num_iterations = num_iterations
    
    
    
    
    reference_type = 'homo'  # no spatial reference for real drums; keep plotting API compatible
    cross_coords = np.array([0, 0, 0])
    cross_ends = np.array([0, 0, 0])

    #mockup_path= f'test/mockups/HOM/drum_bundles/weightcorrection_nodrum2/{mockup_case}'
    #mockup_path= f'test/mockups/HOM/drum_bundles/definitive_recalib_2/{mockup_case}'
    mockup_path = str(input_path) if input_path is not None else f'test/realdrums/{mockup_case}' 
    samples_file = os.path.join(results_path, 'samples_activity.npy') #saves the samples for posterior analysis
     
    mockup =Saruman.load(mockup_path)
    params = init_solver(mockup, isotope_list, exclude_eu154_715=exclude_eu154_715)
    mask = mask_generation(mockup)
    mask_indices = np.where(mask == 1)
    (
        t, eta, w, obs, obs_sigma, peak_to_iso, n_isotopes, selected_isotopes,
        selected_energies, eta_audit, observation_metadata, excluded_gamma_lines
    ) = params

    # Zero observations correspond to peak areas not reported/detected by the
    # spectrum analysis pipeline. For this sensitivity run they are omitted from
    # the likelihood and from the main PPC, but are recorded in an audit CSV.
    obs_np_for_mask = np.asarray(obs)
    used_in_likelihood_np = np.ones(obs_np_for_mask.shape, dtype=bool)
    if omit_zero_observations:
        used_in_likelihood_np &= obs_np_for_mask > 0
    used_in_likelihood = jax.device_put(jnp.asarray(used_in_likelihood_np))
    if not np.any(used_in_likelihood_np):
        raise ValueError("No observations remain in the likelihood after omitting zero-valued observations.")
    I, J, K = [31, 48, 31]  # Dimensions of the voxel grid
    
     
    
    
    # Aquí van tus valores del forward model
     
    
    
    
    
    print("Reconstructing isotopes:", isotope_list)

    """SVI"""
    #from numpyro.infer import init_to_mean,init_to_median,init_to_value,init_to_sample
     
    guide = AutoNormal(forward_model )
    # Initialize SVI
    
    optimizer = Adam(step_size=float(optimization_cfg.get("learning_rate", 0.001)))  # You can tune the step size
    svi = SVI(forward_model, guide, optimizer, loss=Trace_ELBO(num_particles=int(optimization_cfg.get("elbo_particles", 10))))
    
    
    # Run SVI
    # Number of optimization steps
    svi_result = svi.run(
        random.PRNGKey(seed),
        num_iterations,
        obs=obs,
        w=w,
        eta=eta,
        t=t,
        mask=mask,
        peak_to_iso=peak_to_iso,
        n_isotopes=n_isotopes,
        likelihood_mask=used_in_likelihood
    )
    
    # Extract variational parameters
    params = svi_result.params
    elbo_values = svi_result.losses
    elbo_np = np.array(elbo_values)
    
    # Guardar como CSV
    df = pl.DataFrame(elbo_np)
    df.write_csv(os.path.join(results_path, "elbo_values_initial.csv"))
    # ELBO check
    
    import matplotlib.ticker as ticker
    plt.plot(elbo_values)
    plt.yscale("log")
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    # Use scientific notation
    ax = plt.gca()
    
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-1, 1))
    
     
    plt.tight_layout()
    plt.show()
    
    
    """Sample collection""" 
    
    
    
    
    # Posterior predictive check
    predictive = numpyro.infer.Predictive(model=forward_model,guide=guide,num_samples=n_samples)
     
    samples = predictive(random.PRNGKey(1), w=w, eta=eta, t=t, mask=mask, peak_to_iso=peak_to_iso, n_isotopes=n_isotopes, likelihood_mask=used_in_likelihood, obs=None) 
    prior_samples= samples['A']
    # Posterior predictive sampling
    predictive = numpyro.infer.Predictive(
        model=forward_model,  # Forward model
        guide=guide,          # Variational guide
        num_samples=n_samples,     # Number of posterior samples
        params=params         # Fitted parameters from SVI
    )
    
    # Generate posterior predictive samples
    posterior_samples = predictive(
        random.PRNGKey(1), 
        w=w, 
        eta=eta, 
        t=t, 
        mask=mask,
        peak_to_iso=peak_to_iso,
        n_isotopes=n_isotopes
    
    )
    from scipy.stats import pearsonr
    # Extract predictions and observed data
    posterior_predictions_all = posterior_samples['Y_obs']  # Shape: (num_samples, all retained data points)
    posterior_predictions = posterior_predictions_all[:, used_in_likelihood_np]
    observed_data = obs[used_in_likelihood_np]
    observed_sigma_data = obs_sigma[used_in_likelihood_np]
    ppc_by_index(posterior_predictions, observed_data, n_draws=25, title="Posterior predictive (used observations only)")
    A_retrieved =posterior_samples['A']
    magnitude = posterior_samples['magnit']
    Activity_field = jnp.sum(posterior_samples['A_field'],axis=(1,2,3))
    # Compute Pearson correlation on log-transformed values
    log_A = np.log10(Activity_field)
    log_mag = np.log10(magnitude)
    r, p_value = pearsonr(log_A, log_mag)
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(Activity_field, magnitude, alpha=0.5, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('A (log scale)')
    plt.ylabel('Magnitude (log scale)')
    plt.title('Posterior Samples: A vs Magnitude (log-log)')
    corr = np.array(posterior_samples["eff_correction_factors"])
    print("p90:", np.percentile(corr, 90, axis=0))
    print("max:", np.max(corr, axis=0))
    # Annotate the correlation coefficient
    plt.text(
        0.05, 0.95,
        f"$r$ = {r:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )
    
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    # Compute summary statistics for predictions
    posterior_mean = jnp.mean(posterior_predictions, axis=0)
    posterior_std = jnp.std(posterior_predictions, axis=0)

    # Plot observed vs predicted distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(
        posterior_mean, 
        bins=50, 
        kde=True, 
        color="blue", 
        label="Posterior Mean Prediction"
    )
    sns.histplot(
        observed_data, 
        bins=50, 
        kde=True, 
        color="orange", 
        label="Observed Data", 
        alpha=0.7
    )
    plt.axvline(jnp.mean(observed_data), color='red', linestyle='--', label="Observed Mean")
    plt.xlabel("Value")
    plt.ylabel("Density")
    
    plt.legend()
    plt.show()
    
    # Calculate and print discrepancies
    mean_discrepancy = jnp.mean(jnp.abs(posterior_mean - observed_data))
    std_discrepancy = jnp.mean(jnp.abs(posterior_std - jnp.std(observed_data)))
    print(f"Mean Discrepancy: {mean_discrepancy:.4f}")
    print(f"Standard Deviation Discrepancy: {std_discrepancy:.4f}")
    
    # Plot predictive intervals (optional)
    lower = jnp.percentile(posterior_predictions, 2.5, axis=0)
    upper = jnp.percentile(posterior_predictions, 97.5, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(observed_data, label="Observed Data", color="black", linewidth=2)
    plt.fill_between(
        jnp.arange(len(lower)), lower, upper, color="blue", alpha=0.3, label="95% Predictive Interval"
    )
    plt.plot(posterior_mean, label="Posterior Mean", color="blue", linestyle="--")
    plt.xlabel("Detector Index")
    plt.ylabel("Counts")
     
    plt.legend()
     
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()
    
    
    
    
    A_samples = A_retrieved
    A_samples = A_samples.reshape(n_samples,I,J,K)
    A_median = jnp.median(A_samples, axis=0)
    A_mean = jnp.mean(A_samples, axis=0)
     
     
    # Samples are saved once at the end with the other run artifacts
    mean = jnp.mean(posterior_predictions, axis=0)
    median = jnp.median(posterior_predictions, axis=0)
    
    rmse_mean = jnp.sqrt(jnp.mean((mean - observed_data)**2))
    rmse_median = jnp.sqrt(jnp.mean((median - observed_data)**2))
    print("RMSE mean:", float(rmse_mean), "RMSE median:", float(rmse_median))
    from scipy.stats import pearsonr
    log_noise = np.log10(np.array(posterior_samples["noise_check"]))
    r, _ = pearsonr(np.log10(np.array(magnitude)), log_noise)
    print("corr log(magnitude) vs log(noise):", r)
                    
    """Diagnostics"""
    """Posterior Predictive Check"""
    #predictive = numpyro.infer.util.Predictive(forward_model,posterior_samples_ungrouped)
    y_pred_all = predictive(
        random.PRNGKey(1), w=w, eta=eta, t=t, mask=mask, peak_to_iso=peak_to_iso,
        n_isotopes=n_isotopes, likelihood_mask=used_in_likelihood, obs=None
    )['Y_obs']
    # Main PPC is evaluated only on observations that contributed to likelihood.
    posterior_predictive_check(
        y_pred_all, obs, obs_sigma, used_in_likelihood_np, observation_metadata, results_path, 'Yes'
    )
    y_pred_used = y_pred_all[:, used_in_likelihood_np]
    e = observed_data[None, :] - y_pred_used
    rmse = np.sqrt(np.mean(np.asarray(e) ** 2))
    print("PPC RMSE over used observations only:", float(rmse))

    # Save gamma-line and omitted-observation audits for reproducibility.
    eta_audit.write_csv(os.path.join(results_path, "gamma_eta_audit.csv"))
    excluded_gamma_lines.write_csv(os.path.join(results_path, "excluded_gamma_lines.csv"))
    omitted = ~used_in_likelihood_np
    omitted_reasons = np.where(np.asarray(obs)[omitted] == 0, "zero_observation_omitted_from_likelihood", "omitted_from_likelihood")
    omitted_df = pl.DataFrame({
        "isotope": np.asarray(observation_metadata["isotope"])[omitted].tolist(),
        "energy_keV": np.asarray(observation_metadata["energy_keV"])[omitted].tolist(),
        "detector_index_within_peak": np.asarray(observation_metadata["detector_index_within_peak"])[omitted].tolist(),
        "obs": np.asarray(obs)[omitted].tolist(),
        "obs_sigma": np.asarray(obs_sigma)[omitted].tolist(),
        "omission_reason": omitted_reasons.tolist(),
    })
    omitted_df.write_csv(os.path.join(results_path, "omitted_observations_audit.csv"))

    if len(isotope_list) == 1:
        total_recovered_activity((A_retrieved), prior_samples, results_path)
    else:
        total_kBq = np.array(posterior_samples.get("total_activity_kBq", jnp.sum(A_retrieved, axis=(1, 2, 3)) / 1e3))
        fracs = posterior_samples["isotope_fractions_check"]
        isotope_total_samples = posterior_samples.get("isotope_total_activity_kBq", total_kBq[:, None] * fracs)
        rows = []
        for k, iso_name in enumerate(isotope_list):
            iso_total = np.array(isotope_total_samples[:, k])
            rows.append({
                "isotope": iso_name,
                "mean_kBq": float(np.mean(iso_total)),
                "median_kBq": float(np.median(iso_total)),
                "p025_kBq": float(np.percentile(iso_total, 2.5)),
                "p975_kBq": float(np.percentile(iso_total, 97.5)),
                "mean_fraction": float(np.mean(np.array(fracs[:, k]))),
                "median_fraction": float(np.median(np.array(fracs[:, k]))),
            })

            plt.style.use("bmh")
            plt.figure(figsize=(8, 6))
            sns.histplot(iso_total, bins=80, kde=True, stat="density", alpha=0.65, color="blue")
            plt.xlabel(f"Retrieved Activity {iso_name} (kBq)")
            plt.ylabel("Density")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f"total_activity_{iso_name}.jpg"), dpi=300)
            plt.show()
            plt.close()

        # Also save total activity posterior for the shared field.
        plt.style.use("bmh")
        plt.figure(figsize=(8, 6))
        sns.histplot(total_kBq, bins=80, kde=True, stat="density", alpha=0.65, color="blue")
        plt.xlabel("Retrieved total activity (kBq)")
        plt.ylabel("Density")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, "total_activity_joint_total.jpg"), dpi=300)
        plt.show()
        plt.close()

        pl.DataFrame(rows).write_csv(os.path.join(results_path, "joint_activity_summary.csv"))
     
    
    
    
    
    #Plotting
    # Function to plot heatmap with a circle in the xz plane with major and minor ticks, inverted Y-axis
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import LogNorm
     
     
    def plot_xz_heatmap(
        A_mean, circle_radius, x_range, grid_size, major_tick_interval,
        reference_type='point', cross_coords=None, cross_ends=None,# 'cylinder', 'point', or 'both'
    save_path=os.path.join(results_path,"XZ_plane.png")):
     
    
        y_max_idx = jnp.argmax(jnp.mean(A_mean, axis=(0, 2)))  # Max Y index
        A_slice = A_mean[:, y_max_idx, :]  # XZ plane slice
    
        # Reference coordinates (X, Y, Z) 
        #cross_coords = np.array([153.6,201.6,153.6])#Cs137 voxel
        #cross_coords = np.array([96,201.6,-115.2])#Eu152 voxel
        #cross_coords = np.array([-45, 90, -14]) #rod
        #cross_ends = np.array([-10, -90, 150])  #rod 
     
    
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        c = ax.imshow(
            A_slice, cmap='viridis', aspect='auto',
            extent=[x_range[0], x_range[1], x_range[1], x_range[0]],
            norm=LogNorm()
        )
        plt.colorbar(c, label='Activity (Bq)')
    
        # Draw reference circle
        circle = plt.Circle((0, 0), circle_radius, color='black', fill=False, linewidth=2)
        ax.add_artist(circle)
    
        # Rectangle
        if reference_type in ('cylinder', 'both'):
            z, x = cross_coords[2], cross_coords[0]
            ax.plot(z, x, marker='o', color='red', markersize=10, markeredgewidth=2)
    
        # Cross marker
        if reference_type in ('point', 'both'):
            z, x = cross_coords[2], cross_coords[0]
            ax.plot(z, x, marker='x', color='red', markersize=10, markeredgewidth=2)
        if reference_type == 'homo':
            pass
        # Grid and ticks
        major_ticks = np.arange(x_range[0], x_range[1], major_tick_interval)
        minor_ticks = np.arange(x_range[0], x_range[1], grid_size)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, which='both')
        ax.invert_yaxis()
    
        plt.title(f'XZ plane at Y index {y_max_idx}', fontsize=14)
        plt.xlabel('Z (mm)', fontsize=12)
        plt.ylabel('X (mm)', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path,dpi=300)
        plt.show()
    
     
     
     
     
    def plot_xy_heatmap(
        A_mean, rect_height, rect_width, x_range, y_range,
        grid_size, major_tick_interval, reference_type='point', save_path=os.path.join(results_path,"XY_plane.png"),cross_coords=None,cross_ends=None):
        # Rectangle
        thickness = 20
      
        z_max_idx = jnp.argmax(jnp.mean(A_mean, axis=(0, 1)))
        A_slice = A_mean[:, :, z_max_idx]
           
    
     
        # Direction and perpendicular vectors, only needed when a line/cylinder reference is drawn.
        rectangle_coords = None
        if reference_type in ('cylinder', 'both'):
            direction = cross_ends[0:2] - cross_coords[0:2]
            length = np.linalg.norm(direction)
            if length > 0:
                unit_dir = direction / length
                perp = np.array([-unit_dir[1], unit_dir[0]])
                offset = (thickness / 2) * perp
                rectangle_coords = np.array([
                    cross_coords[0:2] + offset,
                    cross_coords[0:2] - offset,
                    cross_ends[0:2] - offset,
                    cross_ends[0:2] + offset
                ])
    
    
        # Extent
        Nx, Ny = A_slice.shape
        dx = (x_range[1] - x_range[0]) / Nx
        dy = (y_range[1] - y_range[0]) / Ny
        extent = [
            x_range[0] - dx / 2, x_range[1] + dx / 2,
            y_range[1] + dy / 2, y_range[0] - dy / 2
        ]
    
        # Plot
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        c = ax.imshow(
            A_slice.T, cmap='viridis', aspect='equal',
            extent=extent, norm=LogNorm()
        )
        cb = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Activity (Bq)', fontsize=12)
       
        if reference_type in ('cylinder', 'both') and rectangle_coords is not None:

            polygon = Polygon(rectangle_coords, closed=True, edgecolor='red', fill=False, linewidth=1.5)
            ax.add_patch(polygon)
    
    
        if reference_type in ('point', 'both'):
            ax.plot(cross_coords[0], cross_coords[1], marker='x', color='red', markersize=10, markeredgewidth=2)
        if reference_type=='homo':
            pass
    
        # Ticks and grid
        def symmetric_ticks(axis_min, axis_max, interval):
            max_abs = max(abs(axis_min), abs(axis_max))
            tick_range = np.arange(-max_abs, max_abs + interval, interval)
            return tick_range[(tick_range >= axis_min) & (tick_range <= axis_max)]
    
        major_ticks_x = symmetric_ticks(*sorted(x_range), major_tick_interval)
        minor_ticks_x = symmetric_ticks(*sorted(x_range), grid_size)
        major_ticks_y = symmetric_ticks(*sorted(y_range), major_tick_interval)
        minor_ticks_y = symmetric_ticks(*sorted(y_range), grid_size)
    
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.4, which='both')
        ax.invert_yaxis()
    
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title(f'XY plane at Z index {z_max_idx}', fontsize=13)
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
            plt.show()
        else:
            plt.show()
    
     
        
    # Define the dimensions for the circle and rectangle    # Add grid
    
    
    """
    #If desired, a 3D render of the activity distribution can be generated uing SimpleITK:
    #import SimpleITK as sitk
       
    sitk_image = sitk.GetImageFromArray(A_mean)
    
    sitk_image.SetOrigin( mockup_drum.mesh.attrs['origin'])
    sitk_image.SetSpacing( mockup_drum.mesh.attrs['spacing'])
    sitk_image.SetDirection( mockup_drum.mesh.attrs['direction'])
    """
     
    
    print(f'Average estimated Activity:  {jnp.mean(jnp.sum(A_samples,axis=(1,2,3)) )/1e3} kBq')
    
    
    """
    This is for checking posterior densities for different results
    """
    
    # z_samples_1 = guide.sample_posterior(random.PRNGKey(10101010), sample_shape=(2000,),params=params)
    # np.save("z_samples_config_good.npy", z_samples_1)
    
    # from numpyro.handlers import block, trace, condition
    
    # def compute_log_pz(model, z_samples, w, eta, t, mask, obs, hide_sites=None):
    #     # inject latent values
    #     conditioned_model = condition(model, data=z_samples)
    #     # hide likelihood if needed
    #     blocked_model = block(conditioned_model, hide=hide_sites or [])
    #     # compute trace
    #     tr = trace(blocked_model).get_trace(w, eta, t, mask, obs=obs)
        
    #     # sum log probs of latent sites
    #     log_pz = sum(
    #         site["fn"].log_prob(site["value"]).sum()
    #         for site in tr.values() if site["type"] == "sample"
    #     )
    #     return log_pz
    
    
    # z_samples_1 = np.load("z_samples_config_good.npy", allow_pickle=True).item()
    # z_samples_2 = np.load("z_samples_config_bad.npy", allow_pickle=True).item()
    # log_pz_list_1 = []
    # log_pz_list_2 = []
    
    # for i in range(2000):
      
    #     single_1 = {k: v[i] for k, v in z_samples_1.items()}
    #     single_1["A_fine"] = single_1['A_fine']*infer_mask.flatten()
    #     single_2 = {k: v[i] for k, v in z_samples_2.items()}
        
    
    #     log_pz_list_1.append(
    #         compute_log_pz(forward_model1, single_1, w, eta, t, mask, obs=obs, hide_sites=["Y_obs"])
    #     )
    #     log_pz_list_2.append(
    #         compute_log_pz(forward_model, single_2, w, eta, t, mask, obs=obs, hide_sites=["Y_obs"])
    #     )
    
    # log_pz_list_1 = np.array(log_pz_list_1)
    # log_pz_list_2 = np.array(log_pz_list_2)
    # import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(8,5))
    # plt.hist(log_pz_list_1, bins=50, alpha=0.5, label="Fixed voxel")
    # plt.hist(log_pz_list_2, bins=50, alpha=0.5, label="Failing model")
    # plt.xlabel("log p(z)")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()
    
    # #summary
    # print("Config 1 mean log p(z):", log_pz_list_1.mean())
    # print("Config 2 mean log p(z):", log_pz_list_2.mean())
    import os
    from datetime import datetime
    import numpy as np
    
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    
    import seaborn as sns
    import polars as pl
    import matplotlib.ticker as ticker
    
    # --- FUNCIONES DE GUARDADO DE RESULTADOS ---
    
    
    def save_run_metadata(results_path, posterior_samples, num_iterations, isotope, omit_zero_observations, exclude_eu154_715, used_in_likelihood_np, excluded_gamma_lines):
        """
        Saves metadata of the real-drum run.
        """
        metadata_file = os.path.join(results_path, "run_metadata.txt")
        
        with open(metadata_file, "w") as f:
            f.write(f"Run Date: {datetime.now()}\n")
            f.write(f"Isotope: {isotope}\n")
            f.write(f"Number of Iterations: {num_iterations}\n\n")
            f.write(f"omit_zero_observations_in_likelihood: {omit_zero_observations}\n")
            f.write(f"exclude_eu154_715_line: {exclude_eu154_715}\n")
            f.write(f"n_retained_observations_before_zero_mask: {int(len(used_in_likelihood_np))}\n")
            f.write(f"n_observations_used_in_likelihood: {int(np.sum(used_in_likelihood_np))}\n")
            f.write(f"n_zero_observations_omitted_from_likelihood: {int(np.sum(~used_in_likelihood_np))}\n")
            f.write(f"n_excluded_gamma_lines: {int(excluded_gamma_lines.height)}\n")
            
            # Recorre los determinísticos que están en posterior_samples
            deterministic_keys = [k for k in posterior_samples.keys() if k not in ["magnit","A_field","A", "Y_obs", "A_coarse", "A_fine"]]
            
            f.write("Forward Model Deterministic Parameters:\n")
            for key in deterministic_keys:
                arr = posterior_samples[key]
            
                # Si arr es escalar (), úsalo tal cual; si no, coge el primer sample
                value = arr if getattr(arr, "ndim", 0) == 0 else arr[0]
            
                # Si value es array con un solo elemento (shape (1,), (1,1), etc), hazlo escalar
                if isinstance(value, jnp.ndarray) and value.size == 1:
                    value = float(jnp.ravel(value)[0])   # seguro para (1,) y también para (1,1)
            
                # Si value es escalar jax, conviértelo a float python (opcional pero útil para escribir)
                elif isinstance(value, jnp.ndarray) and value.ndim == 0:
                    value = float(value)
            
                f.write(f"  {key}: {value}\n")
 
            # Magnitude promedio de los samples (si existe)
            if "magnit" in posterior_samples:
                magnitude_mean = float(jnp.mean(posterior_samples["magnit"]))
                f.write(f"\nMagnitude (mean of posterior samples): {magnitude_mean}\n")
    
    
    def save_elbo(elbo_values, results_path):
        # Convertir a NumPy
        elbo_np = np.array(elbo_values)
        # Asegurarse de que sea 1D o 2D
        elbo_np = elbo_np.reshape(-1, 1)
        
        # Crear el path sin espacios
        elbo_file = os.path.join(results_path, "elbo_values.csv")
        
        # Guardar con Polars
        pl.DataFrame({"ELBO": elbo_np.flatten()}).write_csv(elbo_file)
    def save_samples(A_samples, results_path, mockup_case):
        samples_file = os.path.join(results_path, "samples_activity.npy")
        np.save(samples_file, A_samples)
    
    
     
    save_run_metadata(
        results_path, posterior_samples, num_iterations, isotope,
        omit_zero_observations, exclude_eu154_715,
        used_in_likelihood_np, excluded_gamma_lines
    )
     
     
    save_elbo(elbo_values, results_path)
     
    save_samples(A_samples, results_path, mockup_case)
    
     
    
    circle_radius = 287  # mm
    rect_height = 885  # mm
    rect_width = 287*2  # mm
    grid_size = 19.2  # mm
    x_range = (-288, 288)  # mm for X axis
    y_range = (-451.2, 451.2)  # mm for Y axis
    
    # Example usage:
    
    
    
            
    major_tick_interval = 96  # Define the interval for major ticks (96 mm)
    plot_xz_heatmap(A_mean, circle_radius, x_range, grid_size, major_tick_interval,
                    reference_type=reference_type, cross_coords=cross_coords, cross_ends=cross_ends)
    plot_xy_heatmap(A_mean, rect_height, rect_width, x_range, y_range,
                    grid_size, major_tick_interval,
                    reference_type=reference_type, cross_coords=cross_coords, cross_ends=cross_ends)



def solver_joint(
    isotopes, trial, num_iterations, mockup_case,
    omit_zero_observations=True, exclude_eu154_715=True
):
    """Convenience wrapper for joint multi-isotope inference.

    The results folder naming is unchanged. Running with the same trial name
    overwrites the previous joint outputs in the same path.

    Example:
        solver_joint(['Eu-152', 'Eu-154', 'Ag-110M'], 'real_4_joint', 40000, '1.h5')
    """
    return solver(
        list(isotopes), trial, num_iterations, mockup_case,
        omit_zero_observations=omit_zero_observations,
        exclude_eu154_715=exclude_eu154_715,
    )
