"""Learned state (global vs local scope) and the shared learning ledger.

* :mod:`dt.learn.state`  -- where learned state lives; the single writer of params layers.
* :mod:`dt.learn.ledger` -- append / read / revert ledger entries (``knowledge/ledger.jsonl``).
"""
from dt.learn import ledger, state  # noqa: F401
