"""The always-on service: watch, queue, score, serve.

`src/dfd` is the decision library; this package is the deployment around it.
Nothing here decides anything — it moves files to `dfd.pipeline.decide` and
stores what came back, so that a running system and a correct decision stay
separable concerns.

Deliberately stdlib-only (`http.server`, `sqlite3`, `threading`). This repo
pins every dependency exactly and runs its gates at both ends of every
declared range; a web framework would add that maintenance for a local
service whose whole API is six JSON endpoints and one page.
"""
from __future__ import annotations
