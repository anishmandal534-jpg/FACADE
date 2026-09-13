"""
Instagram public URL scraper for FACADE.

This scraper only attempts to retrieve publicly available page metadata.
It does not use Instagram credentials, cookies, or private-account access.
"""

import json
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


INSTAGRAM_DOMAINS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}


def is_instagram_url(url: str) -> bool:
    """Return True if the URL belongs to Instagram."""

    if not url:
        return False

    try:
        parsed = urlparse(url.strip())
        hostname = (parsed.hostname or "").lower()

        return (
            hostname == "instagram.com"
            or hostname == "www.instagram.com"
            or hostname == "m.instagram.com"
            or hostname.endswith(".instagram.com")
        )

    except Exception:
        return False


def normalize_instagram_url(url: str) -> str:
    """Normalize an Instagram URL."""

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    path = parsed.path.rstrip("/")

    return f"https://www.instagram.com{path}/"


def _clean_text(value):
    """Clean whitespace from extracted text."""

    if not value:
        return ""

    value = str(value)

    value = value.replace("\\n", "\n")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def _get_meta_content(soup, *, property_name=None, name=None):
    """Read a meta tag."""

    tag = None

    if property_name:
        tag = soup.find("meta", attrs={"property": property_name})

    if not tag and name:
        tag = soup.find("meta", attrs={"name": name})

    if not tag:
        return ""

    return _clean_text(tag.get("content", ""))


def _extract_username_from_text(text):
    """
    Try to find an Instagram username from common page text patterns.
    """

    if not text:
        return ""

    patterns = [
        r"instagram\.com/([A-Za-z0-9._]+)/?",
        r"@([A-Za-z0-9._]{1,30})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            username = match.group(1)

            # Avoid accidentally treating generic Instagram paths as usernames.
            blocked = {
                "p",
                "reel",
                "reels",
                "tv",
                "explore",
                "accounts",
                "direct",
                "about",
                "developer",
            }

            if username.lower() not in blocked:
                return username

    return ""


def _extract_json_ld(soup):
    """Extract useful information from JSON-LD blocks."""

    results = []

    for script in soup.find_all(
        "script",
        attrs={"type": "application/ld+json"},
    ):
        raw = script.string or script.get_text()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        if isinstance(data, list):
            results.extend(data)

        elif isinstance(data, dict):
            results.append(data)

    return results


def _extract_embedded_data(soup):
    """
    Look for common embedded JSON blocks.

    Instagram changes its frontend frequently, so this is intentionally
    defensive and only extracts data that is already present in the page.
    """

    found = []

    for script in soup.find_all("script"):

        text = script.string or script.get_text()

        if not text:
            continue

        text = text.strip()

        if len(text) > 5_000_000:
            continue

        if any(
            keyword in text.lower()
            for keyword in [
                "caption",
                "owner",
                "username",
                "display_url",
                "taken_at_timestamp",
            ]
        ):
            found.append(text)

    return found


def _extract_caption_from_text(text):
    """Try to extract a caption from embedded JSON-like text."""

    if not text:
        return ""

    patterns = [
        r'"caption"\s*:\s*"((?:\\.|[^"\\])*)"',
        r'"caption_text"\s*:\s*"((?:\\.|[^"\\])*)"',
        r'"text"\s*:\s*"((?:\\.|[^"\\])*)"',
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in matches:

            try:
                # Decode common JSON escapes.
                decoded = json.loads('"' + match + '"')
            except Exception:
                decoded = match

            decoded = _clean_text(decoded)

            if len(decoded) >= 3:
                return decoded

    return ""


def _extract_username_from_embedded_data(text):
    """Try to extract an owner username from embedded page data."""

    if not text:
        return ""

    patterns = [
        r'"username"\s*:\s*"([^"]+)"',
        r'"owner"\s*:\s*\{[^{}]*"username"\s*:\s*"([^"]+)"',
        r'"user"\s*:\s*\{[^{}]*"username"\s*:\s*"([^"]+)"',
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            return _clean_text(match.group(1))

    return ""


def _extract_timestamp_from_embedded_data(text):
    """Try to extract an Instagram post timestamp."""

    if not text:
        return ""

    patterns = [
        r'"taken_at_timestamp"\s*:\s*(\d+)',
        r'"timestamp"\s*:\s*"([^"]+)"',
        r'"datePublished"\s*:\s*"([^"]+)"',
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = match.group(1)

        # Unix timestamp
        if value.isdigit():

            try:
                timestamp = int(value)

                return datetime.fromtimestamp(
                    timestamp
                ).isoformat()

            except Exception:
                pass

        return _clean_text(value)

    return ""


async def fetch_instagram_page(url: str, timeout: float = 20.0):
    """
    Fetch a publicly accessible Instagram page.

    Returns:
        dict containing status, extracted metadata and diagnostic information.
    """

    if not is_instagram_url(url):
        return {
            "success": False,
            "platform": "Instagram",
            "url": url,
            "error": "The supplied URL is not an Instagram URL.",
        }

    normalized_url = normalize_instagram_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    try:

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:

            response = await client.get(normalized_url)

    except httpx.TimeoutException:

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "error": "Instagram request timed out.",
        }

    except httpx.RequestError as exc:

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "error": f"Instagram request failed: {exc}",
        }

    except Exception as exc:

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "error": f"Unexpected Instagram error: {exc}",
        }

    if response.status_code != 200:

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "status_code": response.status_code,
            "error": (
                f"Instagram returned HTTP "
                f"{response.status_code}."
            ),
        }

    html = response.text

    if not html.strip():

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "error": "Instagram returned an empty page.",
        }

    soup = BeautifulSoup(html, "html.parser")

    # ---------------------------------------------------------
    # OpenGraph metadata
    # ---------------------------------------------------------

    og_title = _get_meta_content(
        soup,
        property_name="og:title",
    )

    og_description = _get_meta_content(
        soup,
        property_name="og:description",
    )

    og_image = _get_meta_content(
        soup,
        property_name="og:image",
    )

    og_url = _get_meta_content(
        soup,
        property_name="og:url",
    )

    og_type = _get_meta_content(
        soup,
        property_name="og:type",
    )

    description = _get_meta_content(
        soup,
        name="description",
    )

    # ---------------------------------------------------------
    # JSON-LD
    # ---------------------------------------------------------

    json_ld_items = _extract_json_ld(soup)

    json_ld_caption = ""
    json_ld_author = ""
    json_ld_date = ""
    json_ld_image = ""

    for item in json_ld_items:

        if not isinstance(item, dict):
            continue

        if not json_ld_caption:

            for key in [
                "caption",
                "description",
                "articleBody",
            ]:
                value = item.get(key)

                if value:
                    json_ld_caption = _clean_text(value)
                    break

        if not json_ld_author:

            author = item.get("author")

            if isinstance(author, dict):
                json_ld_author = _clean_text(
                    author.get("name", "")
                )

            elif isinstance(author, str):
                json_ld_author = _clean_text(author)

        if not json_ld_date:

            for key in [
                "datePublished",
                "uploadDate",
                "dateCreated",
            ]:
                value = item.get(key)

                if value:
                    json_ld_date = _clean_text(value)
                    break

        if not json_ld_image:

            image = item.get("image")

            if isinstance(image, str):
                json_ld_image = image

            elif isinstance(image, list) and image:
                json_ld_image = str(image[0])

            elif isinstance(image, dict):
                json_ld_image = _clean_text(
                    image.get("url", "")
                )

    # ---------------------------------------------------------
    # Embedded page data
    # ---------------------------------------------------------

    embedded_blocks = _extract_embedded_data(soup)

    embedded_caption = ""
    embedded_username = ""
    embedded_timestamp = ""

    for block in embedded_blocks:

        if not embedded_caption:
            embedded_caption = _extract_caption_from_text(
                block
            )

        if not embedded_username:
            embedded_username = (
                _extract_username_from_embedded_data(block)
            )

        if not embedded_timestamp:
            embedded_timestamp = (
                _extract_timestamp_from_embedded_data(block)
            )

        if (
            embedded_caption
            and embedded_username
            and embedded_timestamp
        ):
            break

    # ---------------------------------------------------------
    # Determine best available values
    # ---------------------------------------------------------

    caption = (
        embedded_caption
        or json_ld_caption
        or og_description
        or description
    )

    username = (
        embedded_username
        or _extract_username_from_text(og_title)
        or _extract_username_from_text(og_description)
        or _extract_username_from_text(normalized_url)
    )

    published_at = (
        embedded_timestamp
        or json_ld_date
    )

    image_url = (
        og_image
        or json_ld_image
    )

    canonical_url = (
        og_url
        or normalized_url
    )

    title = og_title or ""

    # ---------------------------------------------------------
    # Determine whether useful content was actually obtained
    # ---------------------------------------------------------

    useful_fields = [
        caption,
        username,
        published_at,
        image_url,
        title,
    ]

    has_useful_content = any(
        bool(str(value).strip())
        for value in useful_fields
    )

    if not has_useful_content:

        return {
            "success": False,
            "platform": "Instagram",
            "url": normalized_url,
            "status_code": response.status_code,
            "error": (
                "Instagram page was reachable, but "
                "no useful public post information "
                "was exposed in the page."
            ),
        }

    # ---------------------------------------------------------
    # Build human-readable text for FACADE
    # ---------------------------------------------------------

    content_parts = [
        "Instagram Public Page",
        f"URL: {canonical_url}",
    ]

    if username:
        content_parts.append(
            f"Username: @{username.lstrip('@')}"
        )

    if title:
        content_parts.append(
            f"Title: {title}"
        )

    if published_at:
        content_parts.append(
            f"Published: {published_at}"
        )

    if caption:
        content_parts.append(
            f"Caption: {caption}"
        )

    if image_url:
        content_parts.append(
            f"Media URL: {image_url}"
        )

    content_text = "\n".join(content_parts)

    return {
        "success": True,
        "platform": "Instagram",
        "url": normalized_url,
        "canonical_url": canonical_url,
        "username": username,
        "title": title,
        "caption": caption,
        "published_at": published_at,
        "image_url": image_url,
        "media_type": og_type or "",
        "content_text": content_text,
        "status_code": response.status_code,
    }


async def scrape_instagram_url(url: str):
    """Convenience wrapper."""

    return await fetch_instagram_page(url)


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python instagram_scraper.py "
            "https://www.instagram.com/p/POST_ID/"
        )
        sys.exit(1)

    test_url = sys.argv[1]

    result = asyncio.run(
        scrape_instagram_url(test_url)
    )

    print("\n" + "=" * 70)
    print("FACADE INSTAGRAM SCRAPER TEST")
    print("=" * 70)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))