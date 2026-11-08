"""Installed SARUMAN console command. Dispatch only; no numerical code."""
import sys


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: saruman {prepare,view,reconstruct,run} [options]')
        print('commands: prepare CT/ASGS package; view HDF5; reconstruct SVI; run both stages')
        print('Run `saruman <command> --help` for details.')
        return
    command = argv[0]
    if command == 'prepare':
        from src.prepare_cli import cli_main
        cli_main()
    elif command in ('view', 'reconstruct'):
        from src.saruman_cli_extra import main as cli_main
        cli_main(argv)
    elif command == 'run':
        from src.saruman_run import main as cli_main
        cli_main(argv[1:])
    else:
        raise SystemExit(f'Unknown command: {command}. Use saruman --help.')


if __name__ == '__main__':
    main()
