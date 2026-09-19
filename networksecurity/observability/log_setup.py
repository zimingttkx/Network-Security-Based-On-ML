"""Log routing for the management plane: level, rotating file and syslog.

config.yaml's ``logging:`` block used to be read by nobody — a configured
level or syslog target silently did nothing.  ``configure_logging`` is the one
place that consumes it, and it is deliberately non-fatal: an unreachable
syslog daemon or an unwritable log directory must not stop the service from
starting, so each sink that fails to initialise is reported and skipped while
the console handler stays in place.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import socket
from pathlib import Path

LOGGER_NAME = "networksecurity"


def _parse_syslog_target(address: str):
    """Split a config value into a SysLogHandler address.

    A filesystem path (``/dev/log`` on Linux, ``/var/run/syslog`` on macOS)
    selects a datagram socket; ``host`` or ``host:port`` selects the network
    form.  Returns None when the value is empty or unusable.
    """
    address = address.strip()
    if not address:
        return None
    if "/" in address:
        return address
    host, _, port = address.partition(":")
    if not host:
        return None
    try:
        return (host, int(port or 514))
    except ValueError:
        logging.getLogger(__name__).warning(
            "logging.syslog_address=%r has an invalid port; syslog disabled", address)
        return None


def _default_syslog_address() -> str:
    """The platform's log socket, when one is actually present."""
    for candidate in ("/dev/log", "/var/run/syslog", "/private/var/run/syslog"):
        if os.path.exists(candidate):
            return candidate
    return ""


def configure_logging(cfg: dict) -> logging.Logger:
    """Attach console / file / syslog handlers to the project logger.

    Idempotent: calling it twice (tests, app re-import) replaces the previous
    handlers instead of doubling every line.
    """
    global _configured

    level_name = str(cfg.get("level") or "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        logging.getLogger(__name__).warning("logging.level=%r is not a known level; using INFO",
                                            cfg.get("level"))
        level = logging.INFO

    root = logging.getLogger(LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_target = str(cfg.get("file") or "").strip()
    if file_target:
        try:
            Path(file_target).expanduser().parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                file_target, maxBytes=int(cfg.get("max_bytes") or 10_485_760),
                backupCount=int(cfg.get("backups") or 5), encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).error("logging.file=%r unusable (%s); "
                                              "continuing with console only",
                                              file_target, exc)

    syslog_address = str(cfg.get("syslog_address") or "").strip() or _default_syslog_address()
    target = _parse_syslog_target(syslog_address)
    if target is not None:
        try:
            # A string address is a filesystem datagram socket; a (host, port)
            # tuple is UDP.  SOCK_DGRAM either way: syslog over TCP is not what
            # the handler's datagram framing assumes.
            local_socket = isinstance(target, str)
            syslog_handler = logging.handlers.SysLogHandler(
                address=target, socktype=socket.SOCK_DGRAM)
            if not local_socket:
                syslog_handler.ident = "nips"
            syslog_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
            root.addHandler(syslog_handler)
        except OSError as exc:
            logging.getLogger(__name__).warning("syslog target %r unreachable (%s); "
                                                "syslog forwarding disabled", target, exc)

    root.setLevel(level)
    root.propagate = False
    return root
