"""A tiny, dependency-free on-disk cache for external bio-database lookups.

Symbol resolution (mygene.info) and interaction search (STRING-db) both hit the
network, and the dashboard calls them repeatedly as the user clicks around the
network / heatmap.  This module memoizes those answers to a per-namespace JSON
file so a repeat lookup is instant and we stay polite to the upstream APIs.

Design notes
------------
* One JSON file per *namespace* (``symbols``, ``interactions``, …), each holding
  ``{key: {"ts": epoch_seconds, "value": <json>}}``.  Small and human-readable.
* Guarded by a process-wide lock because the FastAPI job store runs a small
  thread pool; the cache is read/written from request handler threads.
* Entries carry a timestamp so callers can expire stale answers, but nothing in
  bio-id space changes on human timescales, so the default is "never expire".
* Location is overridable with ``ADGENCOV_CACHE_DIR`` (handy for tests / CI, and
  so a read-only deploy can point it at a writable tmp dir).  All disk errors
  are swallowed — a cache is an optimization, never a correctness dependency.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
# In-memory mirror of each namespace file, lazily loaded once per process.
_MEM: Dict[str, Dict[str, Any]] = {}


def cache_dir() -> str:
    """Return (and create) the directory backing the cache."""
    root = os.environ.get(
        "ADGENCOV_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "adgencov", "bio"),
    )
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        pass
    return root


def _path(namespace: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
    return os.path.join(cache_dir(), f"{safe}.json")


def _load(namespace: str) -> Dict[str, Any]:
    if namespace in _MEM:
        return _MEM[namespace]
    data: Dict[str, Any] = {}
    try:
        with open(_path(namespace), "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
    except (OSError, ValueError):
        data = {}
    _MEM[namespace] = data
    return data


def _flush(namespace: str) -> None:
    try:
        tmp = _path(namespace) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_MEM.get(namespace, {}), fh)
        os.replace(tmp, _path(namespace))
    except OSError:
        pass  # cache write is best-effort


def get(namespace: str, key: str, max_age: Optional[float] = None) -> Optional[Any]:
    """Return a cached value, or ``None`` if absent (or older than *max_age* s)."""
    with _LOCK:
        entry = _load(namespace).get(key)
    if not isinstance(entry, dict) or "value" not in entry:
        return None
    if max_age is not None:
        ts = entry.get("ts", 0)
        if (time.time() - ts) > max_age:
            return None
    return entry["value"]


def set(namespace: str, key: str, value: Any) -> None:  # noqa: A001 - deliberate cache verb
    """Store *value* under (namespace, key) and persist the namespace file."""
    with _LOCK:
        store = _load(namespace)
        store[key] = {"ts": time.time(), "value": value}
        _flush(namespace)


def get_many(namespace: str, keys, max_age: Optional[float] = None):
    """Batch :func:`get`.  Returns ``(hits, misses)`` — a dict and a list."""
    hits: Dict[str, Any] = {}
    misses = []
    for k in keys:
        v = get(namespace, k, max_age=max_age)
        if v is None:
            misses.append(k)
        else:
            hits[k] = v
    return hits, misses


def set_many(namespace: str, items: Dict[str, Any]) -> None:
    """Batch :func:`set` — one file flush for the whole batch."""
    if not items:
        return
    with _LOCK:
        store = _load(namespace)
        now = time.time()
        for k, v in items.items():
            store[k] = {"ts": now, "value": v}
        _flush(namespace)
