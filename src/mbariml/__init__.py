"""mbariml: YOLO detection -> embedding -> clustering -> curation pipeline.

The pipeline is four phases -- Ingest, Enrich, Curate, Emit -- whose
commands all read/write one shared DuckDB database. Every command can be run
on its own, pointed at an existing database, which is what lets you start anywhere -- point an ingest
command (``infer images``/``infer video``) at new data and go straight to
review, or run enrichment over a database any of them produced.

See ``mbariml.cli`` for the unified command line entry point (``mbariml``),
or ``mbariml.steps`` for the individual step implementations.
"""

__version__ = "0.11.0"
