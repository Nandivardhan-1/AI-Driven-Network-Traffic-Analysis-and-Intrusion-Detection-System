"""Entry point so the prototype runs as ``python -m nids <subcommand>``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
