from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import polars as pl
from scipy import interpolate
import os
import joblib
from scipy.interpolate import interp1d
import warnings


# NOTE:
# This global flag is intentionally NOT used inside load_ct2lac_model.
# The effective INPUT_IS_RHO is defined inside load_ct2lac_model().
# Keep this only if other scripts import it, but do not rely on it.
INPUT_IS_RHO = False


def bypass(CT_values, filler_name, filler_density, th_air, th_filler, energy_ref):
    """
    Convierte una lista de valores CT en valores LAC usando thresholds fijos.

    Parameters
    ----------
    CT_values : list or np.ndarray
        Lista o array de valores CT.
    filler_name : str
        Nombre del filler, por ejemplo "Potting soil".
    filler_density : float
        Densidad g/cc del filler.
    th_air : float
        Threshold para separar aire.
    th_filler : float
        Threshold para separar filler.
    energy_ref : float
        Energia de referencia en keV.

    Returns
    -------
    np.ndarray
        Array de LAC values [cm^-1], misma longitud que CT_values.
    """

    filler_mac_df = pl.read_csv((PROJECT_ROOT / f"src/data/mac/{filler_name}.csv"))
    air_mac_df = pl.read_csv((PROJECT_ROOT / "src/data/mac/Air.csv"))
    iron_mac_df = pl.read_csv((PROJECT_ROOT / "src/data/mac/Iron.csv"))

    interpolator_filler = interpolate.interp1d(
        filler_mac_df["E"], filler_mac_df["Mac"], kind="linear", fill_value="extrapolate"
    )
    interpolator_air = interpolate.interp1d(
        air_mac_df["E"], air_mac_df["Mac"], kind="linear", fill_value="extrapolate"
    )
    interpolator_iron = interpolate.interp1d(
        iron_mac_df["E"], iron_mac_df["Mac"], kind="linear", fill_value="extrapolate"
    )

    mac_filler = float(interpolator_filler(energy_ref))
    mac_air = float(interpolator_air(energy_ref))
    mac_iron = float(interpolator_iron(energy_ref))

    densities = {0: 0.001205, 1: float(filler_density), 2: 7.874}

    lac_values_dict = {
        0: mac_air * densities[0],
        1: mac_filler * densities[1],
        2: mac_iron * densities[2],
    }

    arr = np.array(CT_values, dtype=np.float32)

    seg = np.zeros_like(arr, dtype=np.uint8)
    seg[arr <= th_air] = 0
    seg[(arr > th_air) & (arr <= th_filler)] = 1
    seg[arr > th_filler] = 2

    LAC_values = np.zeros_like(arr, dtype=np.float32)
    for label, lac in lac_values_dict.items():
        LAC_values[seg == label] = lac

    return LAC_values



def load_ct2lac_model(debug=False, config=None):
    """
    CT2LAC loader with:
      - MAC interpolator cache per Z
      - rho(mean) interpolation
      - optional PCHIP/log-rho mode
      - optional INPUT_IS_RHO mode
      - one single RHO_SCALE consistently applied to all CT -> rho routes

    Important:
    ----------
    RHO_SCALE is applied in BOTH:
      - get_Z_rho_vectorized()
      - get_lac_vectorized()

    Therefore the fast forward_model route and the direct get_lac_vectorized route
    use the same density scaling.
    """
    import os
    import warnings
    import numpy as np
    import polars as pl
    import joblib
    from scipy.interpolate import interp1d, PchipInterpolator

    # ============================================================
    # User/debug toggles
    # ============================================================
    USE_PCHIP_LOGRHO = False

    # Set True ONLY if mesh.values already stores rho [g/cm3].
    # For normal CT/ADU input, keep False.
    INPUT_IS_RHO_LOCAL = False

    # This is the correction that reproduces the old March-style k*rho test.
    # Set to 1.0 to disable.
    RHO_SCALE = 1.0

    print(
        f"[CT2LAC] USE_PCHIP_LOGRHO={USE_PCHIP_LOGRHO}, "
        f"INPUT_IS_RHO={INPUT_IS_RHO_LOCAL}, RHO_SCALE={RHO_SCALE}"
    )

    # Model/calibration paths come from SARUMAN config, not the current working directory.
    if config is None:
        from src.config import load_config
        config = load_config()
    ct_settings = config["ct2lac"]
    MODEL_PATH = os.fspath(ct_settings["rf_model_path"])
    CALIB_PATH = os.fspath(ct_settings["density_calibration_path"])
    NIST_PATH = os.fspath(ct_settings["nist_database_path"])

    if not all(os.path.exists(p) for p in [MODEL_PATH, CALIB_PATH, NIST_PATH]):
        raise FileNotFoundError(
            f"CT2LAC resource files are missing.\n"
            f"Asegúrate de que existan:\n"
            f" - rf_Z_predictor.joblib\n"
            f" - CT_density_LAC_calibration_data.csv\n"
            f" - LAC_nist_database.csv"
        )

    # ============================================================
    # Load data and model
    # ============================================================
    df_calib = pl.read_csv(CALIB_PATH)
    df_nist = pl.read_csv(NIST_PATH)
    rf_Z = joblib.load(MODEL_PATH)

    Z_available = df_nist["Z"].unique().to_numpy()
    Z_min = int(np.min(Z_available))
    Z_max = int(np.max(Z_available))

    # Cache MAC interpolators per Z
    mac_interp_by_Z = {}
    for Z in Z_available:
        df_element = df_nist.filter(pl.col("Z") == Z)
        mac_interp_by_Z[int(Z)] = interp1d(
            df_element["E"].to_numpy(),
            df_element["Mac"].to_numpy(),
            kind="linear",
            fill_value="extrapolate",
        )

    # ============================================================
    # Density <-> Mean interpolators
    # ============================================================
    cols = df_calib.columns
    if "Material" in cols:
        df_md = df_calib.select(["Material", "Mean", "Density"]).drop_nulls()
        df_md = df_md.filter(pl.col("Material").str.to_lowercase() != "air")
    else:
        df_md = df_calib.select(["Mean", "Density"]).drop_nulls()

    mean = df_md["Mean"].to_numpy().astype(np.float64)
    rho = df_md["Density"].to_numpy().astype(np.float64)

    # Optional physical air anchor, disabled for now to preserve your current behaviour.
    MEAN_AIR = 0.001
    RHO_AIR = 0.001205
    mean = np.concatenate([mean, [MEAN_AIR]])
    rho = np.concatenate([rho, [RHO_AIR]])

    order = np.argsort(mean)
    mean = mean[order]
    rho = rho[order]

    interp_rho_linear = interp1d(mean, rho, kind="linear", fill_value="extrapolate")
    interp_mean_from_density_linear = interp1d(rho, mean, kind="linear", fill_value="extrapolate")

    if USE_PCHIP_LOGRHO:
        rho_pos = np.clip(rho, 1e-6, None)
        pchip_logrho = PchipInterpolator(mean, np.log(rho_pos), extrapolate=True)

        def interp_rho(x):
            x = np.asarray(x, dtype=np.float64)
            return np.exp(pchip_logrho(x))

        if INPUT_IS_RHO_LOCAL:
            ord_r = np.argsort(rho_pos)
            rho_s = rho_pos[ord_r]
            mean_s = mean[ord_r]

            rho_u, idx_start, counts = np.unique(
                rho_s, return_index=True, return_counts=True
            )
            mean_u = np.array(
                [mean_s[i : i + c].mean() for i, c in zip(idx_start, counts)],
                dtype=np.float64,
            )

            pchip_mean_from_rho = PchipInterpolator(rho_u, mean_u, extrapolate=True)

            def interp_mean_from_density(r):
                r = np.asarray(r, dtype=np.float64)
                r = np.clip(r, 1e-6, None)
                return pchip_mean_from_rho(r)
        else:
            def interp_mean_from_density(r):
                raise RuntimeError(
                    "interp_mean_from_density not built because INPUT_IS_RHO=False"
                )
    else:
        def interp_rho(x):
            return interp_rho_linear(np.asarray(x, dtype=np.float64))

        def interp_mean_from_density(r):
            return interp_mean_from_density_linear(np.asarray(r, dtype=np.float64))

    # ============================================================
    # Helper functions
    # ============================================================
    def _rho_from_ct_values(ct_vals):
        """CT/Mean -> rho with the single shared RHO_SCALE."""
        mean_like = np.asarray(ct_vals, dtype=np.float32).ravel()
        rho_est = interp_rho(mean_like).astype(np.float32)
        rho_est = np.clip(RHO_SCALE * rho_est, 0.0, 10.0).astype(np.float32)
        return rho_est, mean_like

    def _rho_from_input_values(vals):
        """Handle either CT input or rho input depending on INPUT_IS_RHO_LOCAL."""
        vals = np.asarray(vals, dtype=np.float32).ravel()

        if INPUT_IS_RHO_LOCAL:
            # If input already is rho, do NOT multiply by RHO_SCALE again.
            # Otherwise you would scale an already physical density field.
            rho_est = np.clip(vals, 0.0, 10.0).astype(np.float32)
            mean_like = interp_mean_from_density(rho_est).astype(np.float32)
            return rho_est, mean_like

        return _rho_from_ct_values(vals)

    def get_rho(LAC_CT):
        # Scalar helper uses same scaling for consistency.
        rho_est, _ = _rho_from_input_values(np.array([LAC_CT], dtype=np.float32))
        return float(rho_est[0])

    def get_Zeff_rf(LAC_CT):
        rho_estimated, mean_like = _rho_from_input_values(np.array([LAC_CT], dtype=np.float32))
        X = np.column_stack([rho_estimated, mean_like]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_pred = int(round(rf_Z.predict(X)[0]))
        return int(np.clip(Z_pred, Z_min, Z_max))

    def get_lac(LAC_CT, E):
        E = np.atleast_1d(E).astype(float)
        rho_estimated = get_rho(LAC_CT)
        Z_assigned = get_Zeff_rf(LAC_CT)

        interp_mac = mac_interp_by_Z[int(Z_assigned)]
        mac_at_E = interp_mac(E)
        lac_estimated = rho_estimated * mac_at_E
        return lac_estimated, Z_assigned

    def get_Z_rho_vectorized(ct_vals):
        """
        Main fast route used by forward_model.py.

        Returns:
        --------
        Z_pred : np.ndarray[int32]
        rho_raw : np.ndarray[float32]
        """
        rho_raw, mean_like = _rho_from_input_values(ct_vals)

        X = np.column_stack([rho_raw, mean_like]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_pred = rf_Z.predict(X).round().astype(np.int32)

        Z_pred = np.clip(Z_pred, Z_min, Z_max).astype(np.int32)
        return Z_pred, rho_raw

    def get_lac_vectorized(CT_values, energy):
        """
        Direct vectorized CT -> LAC route.

        Uses the SAME RHO_SCALE as get_Z_rho_vectorized().
        """
        CT_values = np.atleast_1d(CT_values).astype(np.float32)
        energy = np.atleast_1d(energy).astype(np.float32)

        rho_est, mean_like = _rho_from_input_values(CT_values)

        X_input = np.column_stack([rho_est, mean_like]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_pred = rf_Z.predict(X_input).round().astype(np.int32)

        Z_pred = np.clip(Z_pred, Z_min, Z_max).astype(np.int32)

        LAC_array = np.zeros((len(CT_values), len(energy)), dtype=np.float32)
        for Z in np.unique(Z_pred):
            m = Z_pred == Z
            interp_mac = mac_interp_by_Z[int(Z)]
            LAC_array[m, :] = rho_est[m, None] * interp_mac(energy)
        
        #LAC_array[:] = np.mean(LAC_array, axis=0, keepdims=True)  # ONE-SHOT: homogeneous LAC field for 2_ave.h5
        return np.clip(LAC_array, 0.0, None), Z_pred

    def predict_Z_from_rho_and_ct(rho_vals, ct_vals):
        """
        Optional helper for forward_model mass-calibration experiments.
        Given an already corrected rho and the original CT/Mean values,
        recompute Z with the RF.
        """
        rho_vals = np.asarray(rho_vals, dtype=np.float32).ravel()
        ct_vals = np.asarray(ct_vals, dtype=np.float32).ravel()

        X = np.column_stack([rho_vals, ct_vals]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_pred = rf_Z.predict(X).round().astype(np.int32)

        Z_pred = np.clip(Z_pred, Z_min, Z_max).astype(np.int32)
        return Z_pred

    def debug_plot(LAC_CT, E):
        import matplotlib.pyplot as plt

        lac_est, Z_pred = get_lac(LAC_CT, E)
        rho_est = get_rho(LAC_CT)
        print(f"[DEBUG] LAC_CT={LAC_CT:.4f} -> rho={rho_est:.4f}, Z={Z_pred}")

        plt.figure(figsize=(6, 4))
        plt.plot(E, lac_est, label=f"LAC estimado (Z={Z_pred})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energia [keV]")
        plt.ylabel("mu [1/cm]")
        plt.title(f"Debug LAC predictor (Z={Z_pred})")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.show()

    api = {
        "get_rho": get_rho,
        "get_Zeff_rf": get_Zeff_rf,
        "get_lac": get_lac,
        "get_lac_vectorized": get_lac_vectorized,
        "get_Z_rho_vectorized": get_Z_rho_vectorized,
        "predict_Z_from_rho_and_ct": predict_Z_from_rho_and_ct,
        "mac_interp_by_Z": mac_interp_by_Z,
        "interp_rho_mode": "pchip_logrho" if USE_PCHIP_LOGRHO else "linear",
        "input_is_rho": INPUT_IS_RHO_LOCAL,
        "rho_scale": RHO_SCALE,
        "calib_path": CALIB_PATH,
    }

    if debug:
        api["debug_plot"] = debug_plot

    return api


if __name__ == "__main__":
    api = load_ct2lac_model(debug=True)

    get_lac = api["get_lac"]
    debug_plot = api["debug_plot"]

    E = np.logspace(np.log10(50), np.log10(2000), 200)
    LAC_CT = 14.36
    LAC, Z = get_lac(LAC_CT, E)
    print(f"Z efectivo: {Z}")
    print(f"LAC promedio: {LAC.mean():.5f}")

    debug_plot(LAC_CT, E)
