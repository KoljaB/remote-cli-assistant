from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser


DEFAULT_USER_AGENT = (
    "remote-cli-assistant/1.0 (+https://github.com; local research helper)"
)
TEXTUAL_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/ld+json",
    "application/xhtml+xml",
    "application/xml",
)
SKIP_URL_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".whl",
    ".exe",
    ".msi",
    ".dmg",
    ".deb",
    ".rpm",
)
KEYWORD_HINTS = (
    "install",
    "installation",
    "requirements",
    "dependency",
    "dependencies",
    "readme",
    "setup",
    "quickstart",
    "usage",
    "cuda",
    "torch",
    "python",
    "linux",
    "ollama",
    "lm studio",
)


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    page_excerpt: str = ""
    engine: str = ""


@dataclass
class ResearchPacket:
    provider: str
    reason: str
    query: str
    sources: list[ResearchSource] = field(default_factory=list)

    def urls(self) -> list[str]:
        return [source.url for source in self.sources if source.url]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if tag in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3", "h4", "pre", "code"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3", "h4", "pre", "code"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data.strip():
            return
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


class SearXNGResearchClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int = 20,
        max_results: int = 6,
        fetch_top_n: int = 3,
        excerpt_chars: int = 1200,
        language: str = "all",
        categories: str = "general",
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        cleaned_base_url = base_url.strip().rstrip("/")
        if not cleaned_base_url:
            raise ValueError("A SearXNG base URL is required.")
        self.base_url = cleaned_base_url
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.fetch_top_n = fetch_top_n
        self.excerpt_chars = excerpt_chars
        self.language = language
        self.categories = categories
        self.user_agent = user_agent

    def search(self, *, query: str, reason: str) -> ResearchPacket:
        payload = self._search_payload(query)
        sources = self._collect_sources(payload)
        for source in sources[: self.fetch_top_n]:
            source.page_excerpt = self._fetch_page_excerpt(source.url)
        return ResearchPacket(
            provider="searxng",
            reason=reason,
            query=query,
            sources=sources,
        )

    def _search_payload(self, query: str) -> dict:
        params = {
            "q": query,
            "format": "json",
            "language": self.language,
            "categories": self.categories,
            "safesearch": "0",
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/search?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SearXNG search failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"SearXNG search failed: {exc}") from exc

    def _collect_sources(self, payload: dict) -> list[ResearchSource]:
        results = payload.get("results", [])
        sources: list[ResearchSource] = []
        seen_urls: set[str] = set()
        for item in results:
            url = str(item.get("url") or "").strip()
            if not _is_http_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                ResearchSource(
                    title=_clean_text(str(item.get("title") or "Untitled result")),
                    url=url,
                    snippet=_clean_text(str(item.get("content") or "")),
                    engine=_clean_text(str(item.get("engine") or "")),
                )
            )
            if len(sources) >= self.max_results:
                break
        return sources

    def _fetch_page_excerpt(self, url: str) -> str:
        if url.lower().endswith(SKIP_URL_SUFFIXES):
            return ""
        request = urllib.request.Request(
            url=url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html, text/plain;q=0.9, */*;q=0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if not any(kind in content_type for kind in TEXTUAL_CONTENT_TYPES):
                    return ""
                body = response.read(250_000)
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError:
            return ""
        except urllib.error.URLError:
            return ""

        text = body.decode(charset, errors="replace")
        if "html" in content_type or "xml" in content_type:
            text = _extract_html_text(text)
        return _select_excerpt(text, limit=self.excerpt_chars)


def format_research_packet(packet: ResearchPacket, *, max_sources: int = 5) -> str:
    if not packet.sources:
        return (
            f"Research provider: {packet.provider}\n"
            f"Research reason: {packet.reason}\n"
            f"Search query: {packet.query}\n"
            "Collected sources: none"
        )

    lines = [
        f"Research provider: {packet.provider}",
        f"Research reason: {packet.reason}",
        f"Search query: {packet.query}",
        "Collected sources:",
    ]
    for index, source in enumerate(packet.sources[:max_sources], start=1):
        lines.append(f"{index}. {source.title}")
        lines.append(f"URL: {source.url}")
        if source.engine:
            lines.append(f"Search engine: {source.engine}")
        if source.snippet:
            lines.append(f"Search snippet: {source.snippet}")
        if source.page_excerpt:
            lines.append(f"Page excerpt: {source.page_excerpt}")
    return "\n".join(lines)


def _extract_html_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        return _clean_text(re.sub(r"<[^>]+>", " ", html))
    return _clean_text(extractor.text())


def _select_excerpt(text: str, *, limit: int) -> str:
    normalized = _clean_text(text)
    if len(normalized) <= limit:
        return normalized

    selected_lines: list[str] = []
    seen_lines: set[str] = set()
    for raw_line in normalized.splitlines():
        line = _clean_text(raw_line)
        if not line or line in seen_lines:
            continue
        if len(line) < 35 and not any(keyword in line.lower() for keyword in KEYWORD_HINTS):
            continue
        seen_lines.add(line)
        selected_lines.append(line)
        excerpt = " ".join(selected_lines)
        if len(excerpt) >= limit:
            return excerpt[:limit].rstrip()

    return normalized[:limit].rstrip()


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    cleaned = unescape(without_tags)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
