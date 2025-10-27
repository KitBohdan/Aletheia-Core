"""A minimal stub of the ``httpx`` API sufficient for Starlette's TestClient."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlencode, urljoin, urlparse

from . import _client
from . import _types

__all__ = [
    "BaseTransport",
    "ByteStream",
    "Client",
    "Headers",
    "Request",
    "Response",
    "USE_CLIENT_DEFAULT",
    "_client",
    "_types",
]

USE_CLIENT_DEFAULT = _client.USE_CLIENT_DEFAULT


class Headers:
    """Very small dictionary-like helper for HTTP headers."""

    def __init__(self, data: Optional[Union[Mapping[str, Union[str, Sequence[str]]], Iterable[Tuple[str, str]]]] = None):
        self._items: List[Tuple[str, str]] = []
        if data is None:
            return
        if isinstance(data, Mapping):
            for key, value in data.items():
                self._extend_from_item(key, value)
        else:
            for key, value in data:
                self._items.append((str(key), str(value)))

    def copy(self) -> "Headers":
        return Headers(self._items)

    def _extend_from_item(self, key: str, value: Union[str, Sequence[str]]) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                self._items.append((str(key), str(item)))
        else:
            self._items.append((str(key), str(value)))

    def update(self, other: Union["Headers", Mapping[str, Union[str, Sequence[str]]], Iterable[Tuple[str, str]]]) -> None:
        if isinstance(other, Headers):
            items = other._items
        elif isinstance(other, Mapping):
            items = []
            for key, value in other.items():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        items.append((str(key), str(item)))
                else:
                    items.append((str(key), str(value)))
        else:
            items = [(str(k), str(v)) for k, v in other]
        for key, value in items:
            self._items.append((key, value))

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        key_lower = key.lower()
        for current_key, current_value in reversed(self._items):
            if current_key.lower() == key_lower:
                return current_value
        return default

    def setdefault(self, key: str, value: str) -> str:
        existing = self.get(key)
        if existing is None:
            self._items.append((key, value))
            return value
        return existing

    def multi_items(self) -> List[Tuple[str, str]]:
        return list(self._items)

    def items(self) -> Iterator[Tuple[str, str]]:
        return iter(self._items)

    def __contains__(self, key: str) -> bool:
        key_lower = key.lower()
        return any(current_key.lower() == key_lower for current_key, _ in self._items)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Headers({self._items!r})"


@dataclass(frozen=True)
class URL:
    raw: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_parsed", urlparse(self.raw))

    @property
    def scheme(self) -> str:
        return self._parsed.scheme or "http"

    @property
    def netloc(self) -> bytes:
        return self._parsed.netloc.encode("ascii")

    @property
    def path(self) -> str:
        return self._parsed.path or "/"

    @property
    def raw_path(self) -> bytes:
        path = self._parsed.path or "/"
        query = self._parsed.query
        if query:
            return f"{path}?{query}".encode("ascii", errors="ignore")
        return path.encode("ascii", errors="ignore")

    @property
    def query(self) -> bytes:
        return (self._parsed.query or "").encode("ascii", errors="ignore")

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return self.raw


class Request:
    """Representation of an outbound request passed to the transport."""

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Union[Headers, Mapping[str, Union[str, Sequence[str]]], Iterable[Tuple[str, str]]]] = None,
        content: Optional[Union[str, bytes]] = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url)
        if isinstance(headers, Headers):
            self.headers = headers.copy()
        else:
            self.headers = Headers(headers)
        if content is None:
            self._body = b""
        elif isinstance(content, bytes):
            self._body = content
        else:
            self._body = content.encode("utf-8")

    def read(self) -> bytes:
        return self._body


class ByteStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class Response:
    """Simple container for HTTP responses produced by the transport."""

    def __init__(
        self,
        status_code: int,
        headers: Optional[Iterable[Tuple[str, str]]] = None,
        stream: Optional[ByteStream] = None,
        request: Optional[Request] = None,
    ) -> None:
        self.status_code = int(status_code)
        self.headers = Headers(headers)
        self._stream = stream or ByteStream(b"")
        self.request = request
        self._content: Optional[bytes] = None

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = self._stream.read()
        return self._content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text)

    def read(self) -> bytes:
        return self.content


class BaseTransport:
    """Base transport API used by :class:`Client`."""

    def handle_request(self, request: Request) -> Response:  # pragma: no cover - interface only
        raise NotImplementedError


class Client:
    """Extremely small subset of the ``httpx.Client`` API."""

    def __init__(
        self,
        *,
        base_url: str = "",
        headers: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        transport: Optional[BaseTransport] = None,
        follow_redirects: bool = True,
        cookies: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.base_url = base_url or ""
        self.follow_redirects = follow_redirects
        self._transport = transport or BaseTransport()
        self.headers = Headers(headers)
        self.cookies: Dict[str, str] = dict(cookies or {})

    # Helper methods -----------------------------------------------------------------
    def _merge_url(self, url: Union[str, bytes]) -> str:
        if isinstance(url, bytes):
            url = url.decode("utf-8")
        parsed = urlparse(str(url))
        if parsed.scheme:
            return str(url)
        base = self.base_url or ""
        if not base:
            return str(url)
        return urljoin(base if base.endswith("/") else base + "/", str(url))

    def _prepare_headers(
        self, headers: Optional[Mapping[str, Union[str, Sequence[str]]]]
    ) -> Headers:
        combined = self.headers.copy()
        if headers:
            combined.update(headers)
        if self.cookies and "cookie" not in combined:
            cookie_value = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
            combined.update({"cookie": cookie_value})
        return combined

    def _prepare_content(
        self,
        *,
        content: Optional[Union[str, bytes]] = None,
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None,
        json_data: Any = None,
        headers: Headers,
    ) -> bytes:
        if json_data is not None:
            headers.setdefault("content-type", "application/json")
            body = json.dumps(json_data, ensure_ascii=False).encode("utf-8")
            return body
        if data is not None:
            if isinstance(data, Mapping):
                headers.setdefault("content-type", "application/x-www-form-urlencoded")
                return urlencode({k: str(v) for k, v in data.items()}).encode("utf-8")
            if isinstance(data, str):
                return data.encode("utf-8")
            return data
        if content is None:
            return b""
        if isinstance(content, bytes):
            return content
        return content.encode("utf-8")

    # Public API ---------------------------------------------------------------------
    def request(
        self,
        method: str,
        url: Union[str, bytes],
        *,
        content: Optional[Union[str, bytes]] = None,
        data: Optional[Union[Mapping[str, Any], bytes, str]] = None,
        files: Any = None,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: Any = USE_CLIENT_DEFAULT,
        timeout: Any = USE_CLIENT_DEFAULT,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> Response:
        del files, cookies, auth, follow_redirects, timeout, extensions  # Unused in the stub
        merged_url = self._merge_url(url)
        if params:
            query = urlencode({k: str(v) for k, v in params.items()}, doseq=True)
            separator = "&" if urlparse(merged_url).query else "?"
            merged_url = f"{merged_url}{separator}{query}" if query else merged_url
        prepared_headers = self._prepare_headers(headers)
        body = self._prepare_content(content=content, data=data, json_data=json, headers=prepared_headers)
        request = Request(method, merged_url, headers=prepared_headers, content=body)
        return self._transport.handle_request(request)

    def get(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def options(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("HEAD", url, **kwargs)

    def post(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: Union[str, bytes], **kwargs: Any) -> Response:
        return self.request("DELETE", url, **kwargs)

    # Context management -------------------------------------------------------------
    def close(self) -> None:  # pragma: no cover - nothing to release in the stub
        return None

    def __enter__(self) -> "Client":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()


# Backwards compatibility helpers ----------------------------------------------------
# Re-export the placeholder typing aliases to mirror the real library's structure.
_types = _types
