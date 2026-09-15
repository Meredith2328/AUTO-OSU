"""Entry point for the packaged executable: GUI by default, CLI when arguments are given.

    AUTO-OSU.exe                 -> opens the window
    AUTO-OSU.exe song.mp3 -d Hard -> command line, same options as `python -m autoosu`
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from autoosu.cli import main

    sys.exit(main())
