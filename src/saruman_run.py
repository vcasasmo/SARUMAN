"""End-to-end SARUMAN CLI orchestration; numerical algorithms remain untouched."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="saruman run",
        description="Prepare a CT/ASGS HDF5 package, then reconstruct selected isotopes.",
    )
    parser.add_argument("--ct", required=True, help="CT reconstruction (.mhd)")
    parser.add_argument("--spectra", required=True, help="ASGS spectrum directory")
    parser.add_argument("--isotopes", nargs="+", required=True, help="Isotopes to reconstruct (e.g. Eu-152 Cs-137)")
    parser.add_argument("--package", required=True, help="Intermediate HDF5 package path")
    parser.add_argument("--output", required=True, help="Reconstruction results directory")
    parser.add_argument("--detector", help="Detector positions CSV")
    parser.add_argument("--config", help="YAML configuration")
    parser.add_argument("--iterations", type=int, help="SVI iteration count")
    parser.add_argument("--samples", type=int, help="Posterior sample count")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--trial", help="Run label")
    parser.add_argument("--beta-alpha", type=float)
    parser.add_argument("--beta-beta", type=float)
    parser.add_argument("--magnitude-lower", type=float)
    parser.add_argument("--magnitude-upper", type=float)
    parser.add_argument("--noise-shape", type=float)
    parser.add_argument("--noise-rate", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate preparation inputs; do not compute or write files")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting package and reusing nonempty result directory")
    args = parser.parse_args(argv)

    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")

    root = Path(__file__).resolve().parents[1]
    package = Path(args.package).expanduser().resolve()
    if package.suffix.lower() != '.h5':
        package = package.with_name(package.name + '.h5')
    output = Path(args.output).expanduser().resolve()
    if package.is_dir():
        parser.error(f"Package path is a directory: {package}")
    if output.exists() and not output.is_dir():
        parser.error(f"Results path is not a directory: {output}")
    if not args.dry_run and not args.overwrite:
        if package.exists():
            parser.error(f"Package already exists: {package} (use --overwrite to replace it)")
        if output.exists() and any(output.iterdir()):
            parser.error(f"Results directory is not empty: {output} (use --overwrite to reuse)")

    prepare = [sys.executable, '-m', 'src.prepare_cli', 'prepare',
               '--ct', args.ct, '--spectra', args.spectra, '--output', str(package)]
    reconstruct = [sys.executable, '-m', 'src.saruman_cli_extra', 'reconstruct',
                   '--input', str(package), '--isotopes', *args.isotopes,
                   '--output', str(output)]
    for name in ('detector', 'config'):
        value = getattr(args, name)
        if value is not None:
            prepare.extend([f'--{name}', value])
            reconstruct.extend([f'--{name}', value])
    for name in ('iterations', 'samples', 'seed', 'trial',
                 'beta_alpha', 'beta_beta', 'magnitude_lower', 'magnitude_upper',
                 'noise_shape', 'noise_rate'):
        value = getattr(args, name)
        if value is not None:
            reconstruct.extend([f'--{name.replace("_", "-")}', str(value)])
    if args.overwrite:
        prepare.append('--overwrite')
        reconstruct.append('--overwrite')
    if args.dry_run:
        prepare.append('--dry-run')

    print('SARUMAN: "One path, from the seeing stone to the posterior."', flush=True)
    try:
        subprocess.run(prepare, check=True)
        if args.dry_run:
            print(f'Next step (not executed): reconstruct {", ".join(args.isotopes)} from {package} into {output}')
            print('DRY RUN: CT, surrogate, SVI, and HDF5 writing were skipped.')
        else:
            subprocess.run(reconstruct, check=True)
            print(f'SARUMAN: "The work is complete." Package: {package}; results: {output}', flush=True)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode or 1, f'SARUMAN: stage failed with exit status {exc.returncode}.\n')


if __name__ == '__main__':
    main()
