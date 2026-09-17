"""mbariml: YOLO detection -> embedding -> clustering -> curation pipeline.

The pipeline is four phases -- Ingest, Enrich, Curate, Emit -- whose
commands all read/write one shared DuckDB database. Every command can be run
on its own, pointed at an existing database, which is what lets you start anywhere -- point an ingest
command (``infer images``/``infer video``) at new data and go straight to
review, or run enrichment over a database any of them produced.

See ``mbariml.cli`` for the unified command line entry point (``mbariml``),
or ``mbariml.steps`` for the individual step implementations.
"""

# Read from the installed package metadata rather than hardcoded here.
# This was a second, hand-maintained copy of pyproject.toml's version and it
# had silently drifted three releases behind (0.13.0 vs 0.16.1) -- and it is
# not decorative: `export id` stamps it into every .id file as
# `# generator: mbariml vX.Y.Z`, so every sidecar written in between claimed
# a version that had not produced it. Provenance nobody updates is
# provenance that lies.
from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("mbariml")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"
