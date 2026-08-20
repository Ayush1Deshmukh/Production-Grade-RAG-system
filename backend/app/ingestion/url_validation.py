"""
app/ingestion/url_validation.py
───────────────────────────────
Guards the URLs that /ingest is willing to fetch.

The endpoint makes the *server* issue the request, so an unchecked URL lets a
caller reach anything the server can reach — cloud metadata endpoints
(169.254.169.254), localhost admin ports, private subnets. Every URL therefore
has to be http(s), resolve exclusively to public addresses, and — when an
allowlist is configured — sit under one of the permitted hostnames.
"""

import ipaddress
import logging
import socket
from typing import List
from urllib.parse import urlparse

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_SCHEMES = {"http", "https"}


class UrlNotAllowed(ValueError):
    """Raised when a requested URL fails validation."""


def _resolved_addresses(hostname: str) -> List[str]:
    """Every IP the hostname resolves to (v4 and v6)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UrlNotAllowed(f"Hostname could not be resolved: {hostname}") from exc
    return sorted({info[4][0] for info in infos})


def _is_public(address: str) -> bool:
    """
    True only for globally routable addresses.

    `is_global` already excludes loopback, link-local (including the cloud
    metadata range), private, reserved, and multicast space.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def _host_allowed(hostname: str, allowlist: List[str]) -> bool:
    """Exact hostname match, or a subdomain of an allowlisted host."""
    host = hostname.lower().rstrip(".")
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowlist)


def validate_ingest_url(url: str, allowlist: List[str] | None = None) -> None:
    """Raise UrlNotAllowed if `url` is not safe for the server to fetch."""
    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlNotAllowed(f"Only http/https URLs may be ingested, got: {url!r}")

    hostname = parsed.hostname
    if not hostname:
        raise UrlNotAllowed(f"URL has no hostname: {url!r}")

    allowlist = settings.ingest_domain_allowlist if allowlist is None else allowlist
    if allowlist and not _host_allowed(hostname, allowlist):
        raise UrlNotAllowed(f"Host {hostname!r} is not in the ingest allowlist")

    # An allowlisted hostname can still point at a private address, so the
    # resolution check runs regardless of the allowlist.
    addresses = _resolved_addresses(hostname)
    private = [addr for addr in addresses if not _is_public(addr)]
    if private:
        raise UrlNotAllowed(
            f"Host {hostname!r} resolves to non-public address(es) {private} — refusing to fetch"
        )


def validate_ingest_urls(urls: List[str], allowlist: List[str] | None = None) -> None:
    """Validate every URL, reporting all failures at once."""
    problems = []
    for url in urls:
        try:
            validate_ingest_url(url, allowlist)
        except UrlNotAllowed as exc:
            problems.append(str(exc))
    if problems:
        raise UrlNotAllowed("; ".join(problems))
