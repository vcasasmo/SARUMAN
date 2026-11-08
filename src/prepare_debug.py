"""Spatial MetaImage diagnostic exports during SARUMAN prepare only.

All input volumes use mesh axis order (x,y,z). SimpleITK expects (z,y,x).
The attenuation export is optical depth tau BEFORE transmission=exp(-tau).
The efficiency export is the surrogate mean (last index 0), not its variance.
"""
from pathlib import Path
import numpy as np


def _settings(batch):
    config = getattr(batch, "config", None) or {}
    return config.get("debug", {})


def _positions(batch, settings):
    total = len(batch.detector_positions)
    requested = settings.get("positions", "auto")
    if requested == "auto":
        count = min(int(settings.get("max_positions", 4)), total)
        if count < 1:
            raise ValueError("debug.max_positions must be positive")
        return set(int(x) for x in np.linspace(0, total - 1, count).round())
    if not isinstance(requested, list) or any(type(x) is not int for x in requested):
        raise ValueError("debug.positions must be 'auto' or a list of zero-based integers")
    if any(pos < 0 or pos >= total for pos in requested):
        raise ValueError(f"debug.positions must be between 0 and {total - 1}")
    return set(requested)


def should_export_volume(batch, position, energy_index, kind):
    settings = _settings(batch)
    if not settings.get("enabled", False):
        return False
    flag = {"ct": "export_coarsened_ct", "attenuation": "export_attenuation",
            "efficiency": "export_efficiency"}[kind]
    if not settings.get(flag, False):
        return False
    if kind == "ct":
        return True
    target_energy_idx = int(settings.get("energy_index", 0))
    n_energies = len(batch.energies_detected)
    if target_energy_idx < 0 or target_energy_idx >= n_energies:
        raise ValueError(f"debug.energy_index must be between 0 and {n_energies - 1}")
    return energy_index == target_energy_idx and position in _positions(batch, settings)


def export_prepare_volume(batch, kind, volume_xyz, position=None, energy_index=None):
    import SimpleITK as sitk

    settings = _settings(batch)
    output_dir = Path(settings.get("output_dir", "debug")).expanduser()
    # Relative debug directory follows the caller's working directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    if position is None:
        name = kind
    else:
        name = f"{kind}_pos_{position:03d}_energy_{energy_index:03d}"

    data = np.asarray(volume_xyz, dtype=np.float32)
    if data.ndim != 3 or data.shape != batch.mesh.shape:
        raise ValueError(f"Debug volume {name}: expected mesh shape {batch.mesh.shape}, got {data.shape}")
    image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    image.SetOrigin(tuple(float(x) for x in batch.mesh.attrs["origin"]))
    image.SetSpacing(tuple(float(x) for x in batch.mesh.attrs["spacing"]))
    image.SetDirection(tuple(float(x) for x in batch.mesh.attrs["direction"]))
    path = output_dir / f"{name}.mhd"
    sitk.WriteImage(image, str(path))
    print(f"[PREPARE DEBUG] {path} (with companion raw data)")
