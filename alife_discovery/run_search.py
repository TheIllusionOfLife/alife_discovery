"""Backward-compatibility shim: CLI entrypoint moved to experiments/search.py."""

from alife_discovery.experiments.search import main as main

if __name__ == "__main__":
    main()
