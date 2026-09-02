"""mbariml: YOLO detection -> embedding -> clustering -> curation pipeline.

The pipeline is a chain of independent steps that all read/write a shared
DuckDB database. Every step can be run on its own, pointed at an existing
database, which is what lets you "start at any step" (including step 8,
standalone inference on a new batch of images).

See ``mbariml.cli`` for the unified command line entry point (``mbariml``),
or ``mbariml.steps`` for the individual step implementations.
"""

__version__ = "0.10.0"
