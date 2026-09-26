import multiprocessing
import sys

from src.cli import main as cli_main
from src.desktop import main as desktop_main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(cli_main(sys.argv[2:]))
    desktop_main()
