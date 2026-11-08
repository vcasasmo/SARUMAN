"""SARUMAN metadata inspection and Bayesian SVI reconstruction commands.

Metadata mode is lazy: inspect dataset shapes/metadata without materializing
1.5 GB HDF5 transmission or efficiency datasets.
"""
import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inspect(path, *, verbose=True):
    import h5py
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 package not found: {path}")
    with h5py.File(path, 'r') as h5:
        if verbose:
            print(f'SARUMAN: "The seeing stone reveals the contents of {path.name}."')
            print(f'Package: {path} ({path.stat().st_size / 2**30:.2f} GiB)')
            print('Datasets (shape, dtype):')
            for key in sorted(h5.keys()):
                obj = h5[key]
                if isinstance(obj, h5py.Dataset):
                    print(f'  {key:27s} {str(obj.shape):20s} {obj.dtype}')
        required = ('mesh_values', 'spectrum_analyzed', 'spc_time', 'detector_efficiencies',
                    'transmission', 'energies_detected', 'isotopes_detected')
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f'Incomplete package; missing datasets: {", ".join(missing)}')
        raw_iso = h5['isotopes_detected'].asstr()[()]
        isotopes = [str(v) for v in raw_iso]
        energies = [float(v) for v in h5['energies_detected'][()]]
        if len(isotopes) != len(energies):
            raise ValueError('Isotope and energy datasets have unequal lengths')
        if verbose:
            print('Gamma lines (keV):')
            for isotope, energy in zip(isotopes, energies):
                print(f'  {isotope:12s} {energy:.3f}')
            print(f'Detected isotope names: {", ".join(dict.fromkeys(isotopes))}')
            print('Metadata inspection complete; large arrays were not loaded.')
    return path, isotopes


def reconstruct(args):
    path, detected = inspect(args.input, verbose=False)
    isotopes = list(dict.fromkeys(args.isotopes))
    missing = [name for name in isotopes if name not in detected]
    if missing:
        raise ValueError(f'Isotope(s) not found in package: {", ".join(missing)}; available: {", ".join(dict.fromkeys(detected))}')
    from src.config import load_config
    cfg = load_config(args.config)
    solver_cfg = cfg['solver']
    if args.iterations is not None:
        solver_cfg['iterations'] = args.iterations
    if args.samples is not None:
        solver_cfg['posterior_samples'] = args.samples
    if args.seed is not None:
        solver_cfg['seed'] = args.seed
    for attr, setting in (('beta_alpha', 'beta_alpha'), ('beta_beta', 'beta_beta'),
                          ('magnitude_lower', 'magnitude_exponent_lower'),
                          ('magnitude_upper', 'magnitude_exponent_upper'),
                          ('noise_shape', 'noise_gamma_shape'), ('noise_rate', 'noise_gamma_rate')):
        value = getattr(args, attr)
        if value is not None:
            solver_cfg['prior'][setting] = value
    if solver_cfg['prior']['magnitude_exponent_lower'] >= solver_cfg['prior']['magnitude_exponent_upper']:
        raise ValueError('Magnitude lower bound must be smaller than upper bound')
    for key in ('beta_alpha', 'beta_beta', 'noise_gamma_shape', 'noise_gamma_rate'):
        if solver_cfg['prior'][key] <= 0:
            raise ValueError(f'Prior {key} must be positive')
    if args.detector:
        cfg['paths']['detector_coordinates_file'] = Path(args.detector).expanduser().resolve()
    detector = cfg['paths']['detector_coordinates_file']
    if not Path(detector).is_file():
        raise FileNotFoundError(f'Detector coordinate CSV not found: {detector}')
    if args.output:
        results = Path(args.output).expanduser().resolve()
    else:
        label = '__'.join(isotopes).replace('/', '-')
        results = (Path.cwd() / 'results' / f'{path.stem}_{label}_trial_{args.trial}').resolve()
    if results.exists() and any(results.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Results directory not empty: {results} (use --overwrite to reuse)')
    if args.dry_run:
        print(f'Input: {path}')
        print(f'Isotopes: {", ".join(isotopes)}')
        print(f'Iterations: {solver_cfg["iterations"]}; posterior samples: {solver_cfg["posterior_samples"]}')
        print(f'Detector positions: {detector}')
        print(f'Results: {results}')
        print('DRY RUN: SVI not started; HDF5 arrays not loaded.')
        return
    results.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from src.solver import solver
    print(f'SARUMAN: "Let the posterior speak." Input: {path}', flush=True)
    solver(isotopes, args.trial, solver_cfg['iterations'], path.name,
           omit_zero_observations=solver_cfg['observations']['omit_zero'],
           input_path=path, output_dir=results, config=cfg,
           detector_csv=detector)
    print(f'SARUMAN: "The posterior has spoken." Results: {results}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='saruman')
    commands = parser.add_subparsers(dest='command', required=True)
    view = commands.add_parser('view', help='Inspect HDF5 metadata without reading large arrays')
    view.add_argument('--input', required=True, help='Prepared drum HDF5')
    recon = commands.add_parser('reconstruct', help='Bayesian SVI reconstruction from HDF5')
    recon.add_argument('--input', required=True, help='Prepared drum HDF5')
    recon.add_argument('--isotopes', nargs='+', required=True, metavar='ISOTOPE', help='Names exactly as shown by saruman view')
    recon.add_argument('--output', help='Results directory (default: ./results/<package>_<isotopes>_trial_<trial>)')
    recon.add_argument('--config', help='Override YAML configuration')
    recon.add_argument('--detector', help='Detector coordinate CSV (default: config.yaml)')
    recon.add_argument('--iterations', type=int, help='SVI steps')
    recon.add_argument('--samples', type=int, help='Posterior sample count')
    recon.add_argument('--seed', type=int, help='SVI random seed')
    recon.add_argument('--trial', default='1', help='Run label')
    recon.add_argument('--beta-alpha', type=float)
    recon.add_argument('--beta-beta', type=float)
    recon.add_argument('--magnitude-lower', type=float)
    recon.add_argument('--magnitude-upper', type=float)
    recon.add_argument('--noise-shape', type=float)
    recon.add_argument('--noise-rate', type=float)
    recon.add_argument('--dry-run', action='store_true', help='Validate input and settings without inference')
    recon.add_argument('--overwrite', action='store_true', help='Allow reuse of existing nonempty results directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'view':
            inspect(args.input)
        elif args.command == 'reconstruct':
            if args.iterations is not None and args.iterations < 1:
                raise ValueError('iterations must be positive')
            if args.samples is not None and args.samples < 1:
                raise ValueError('samples must be positive')
            reconstruct(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'SARUMAN ERROR: {exc}\n')


if __name__ == '__main__':
    main()
