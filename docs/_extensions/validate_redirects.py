"""Sphinx extension that warns about broken redirect targets at build time.

Hooks into the ``build-finished`` event and checks every redirect target
against ``env.found_docs``.  Broken targets are emitted as Sphinx warnings,
which means ``-W`` (warnings-as-errors) will fail the build.
"""

from __future__ import annotations

import logging
from typing import Any

from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


def _validate(app: Sphinx, exception: Exception | None) -> None:
    if exception:
        return

    redirects: dict[str, str] = getattr(app.config, "redirects", None) or {}
    if not redirects:
        return

    found_lower = {doc.lower() for doc in app.env.found_docs}

    broken: list[tuple[str, str, str]] = []
    for source, target in redirects.items():
        if target.startswith(("http://", "https://")):
            continue

        docname = target.removesuffix(".html")
        if docname.lower() in found_lower:
            continue
        if docname.lower().endswith("/index") and docname.lower().removesuffix("/index") in found_lower:
            continue

        broken.append((source, target, docname))

    if broken:
        logger.warning(
            "[validate_redirects] %d broken redirect target(s):", len(broken)
        )
        for source, target, docname in broken:
            logger.warning("  %s → %s  (docname %r not in found_docs)", source, target, docname)


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("build-finished", _validate)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
