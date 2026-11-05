"""Minimal client for the public Binance data archive (data.binance.vision): listing, verified download, parsing.

Every zip is checked against its published ``.CHECKSUM`` (SHA-256); unverifiable files are refused.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
import time
import zipfile

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

_local = threading.local()


class ArchiveError(RuntimeError):
    pass


def session() -> requests.Session:
    """One HTTP session per thread."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def list_prefixes(prefix: str) -> list[str]:
    """Child "directories" under ``prefix`` (paginated S3 listing)."""
    return _list(prefix, "Prefix")


def list_keys(prefix: str) -> list[str]:
    """Object keys under ``prefix`` (paginated S3 listing)."""
    return _list(prefix, "Key")


def download_verified(path: str, retries: int = 4, timeout: float = 30.0) -> bytes | None:
    """Bytes of ``BASE_URL/path`` verified against ``path.CHECKSUM``; None if the file is not published."""
    payload = _get(f"{BASE_URL}/{path}", retries, timeout)
    if payload is None:
        return None
    checksum = _get(f"{BASE_URL}/{path}.CHECKSUM", retries, timeout)
    if checksum is None:
        raise ArchiveError(f"{path} has no published .CHECKSUM; refusing unverified data")
    expected = checksum.decode("utf-8", errors="replace").split()[0].strip().lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ArchiveError(f"SHA-256 mismatch for {path}: published {expected}, downloaded {actual}")
    return payload


def read_zip_csv(payload: bytes, columns: list[str]) -> pd.DataFrame:
    """The single CSV inside an archive zip. Older files have no header row; newer ones do."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("utf-8")
    first = text.split("\n", 1)[0]
    has_header = not first[:1].isdigit()
    frame = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    if not has_header:
        frame.columns = columns[: frame.shape[1]]
    return frame


def to_utc(values: pd.Series) -> pd.Series:
    """Millisecond epochs to UTC timestamps; microsecond epochs are detected and converted too."""
    numeric = pd.to_numeric(values, errors="coerce")
    unit = "us" if numeric.dropna().gt(1e14).any() else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def _list(prefix: str, tag: str) -> list[str]:
    items: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        response = session().get(LIST_URL, params=params, timeout=60)
        response.raise_for_status()
        text = response.text
        found = re.findall(rf"<{tag}>([^<]*)</{tag}>", text)
        items.extend(item for item in found if item != prefix)
        if "<IsTruncated>true</IsTruncated>" not in text:
            return items
        next_marker = re.search(r"<NextMarker>([^<]*)</NextMarker>", text)
        marker = next_marker.group(1) if next_marker else (found[-1] if found else "")
        if not marker:
            return items


def _get(url: str, retries: int, timeout: float) -> bytes | None:
    for attempt in range(retries):
        try:
            response = session().get(url, timeout=timeout)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 200:
            return response.content
        time.sleep(1.5 * (attempt + 1))
    raise ArchiveError(f"giving up on {url} after {retries} attempts")
