import os
import json
import shutil
import io
import base64
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from .ingestion import process_and_store_document
from .embeddings import embed_text
from .graph_db import graph_db
from .vector_db import client as db_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pypdf
import PIL.Image
import re
from urllib.parse import urljoin, urlparse, unquote, quote
import httpx
from bs4 import BeautifulSoup
from twikit import Client

# Resolve all backend-owned paths relative to backend/main.py.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Load the project-root .env as the authoritative configuration.
# Then load backend/.env only for variables that are not already defined.
# This prevents a stale backend/.env from silently overriding a valid root key.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)


app = FastAPI()

# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS=https://your-app.vercel.app,https://your-custom-domain.com
# Falls back to "*" (allow any origin) if not set, which is fine for a
# public read-mostly demo but should be tightened once the Vercel URL
# is known. NOTE: allow_credentials must be False when allow_origins is
# "*" -- browsers reject a credentialed wildcard response, and this app
# does not rely on cookies for auth, so False is safe here.
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel's deployed filesystem is read-only.
# /tmp is the writable directory in Vercel serverless functions.
UPLOAD_DIR = os.path.join("/tmp", "facade_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenRouter configuration
# Prefer OPENROUTER_API_KEY. DEEPSEEK_API_KEY is kept as a backward-compatible
# fallback so an existing deployment does not break.
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or os.getenv("DEEPSEEK_API_KEY", "").strip()
)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free"
).strip()

# OpenRouter model-level fallbacks. OpenRouter will try these in order if the
# primary model/provider is rate-limited, unavailable, or otherwise fails.
# Keep the list short so requests do not spend excessive time failing over.
OPENROUTER_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "google/gemma-4-31b-it:free,"
        "nex-agi/nex-n2.5-pro:free"
    ).split(",")
    if m.strip()
]

# Vision-capable models are kept separate because normal text models may not
# accept image_url content.
OPENROUTER_VISION_MODEL = os.getenv(
    "OPENROUTER_VISION_MODEL",
    "inclusionai/ling-3.0-flash-vl:free"
).strip()
OPENROUTER_VISION_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_VISION_FALLBACK_MODELS",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,"
        "google/gemma-4-31b-it:free"
    ).split(",")
    if m.strip()
]

# The old openrouter/free router has been unreliable in this deployment.
# If it is still present in .env, use a concrete model instead.
if OPENROUTER_MODEL == "openrouter/free":
    OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

deepseek_client = None

if OPENROUTER_API_KEY:
    deepseek_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL",
                "http://127.0.0.1:8000"
            ),
            "X-Title": os.getenv(
                "OPENROUTER_APP_NAME",
                "FACADE Persona Twin"
            ),
        },
    )
else:
    print(
        "WARNING: OPENROUTER_API_KEY is not set. "
        "The backend will start, but LLM/vision endpoints will return a clear "
        "configuration error until the key is added to .env."
    )


def _key_fingerprint(key: str) -> str:
    """Safe diagnostic identifier; never prints the full API key."""
    key = (key or "").strip()
    if len(key) < 8:
        return "NOT_SET"
    return f"{key[:12]}...{key[-4:]}"


print(
    f"[OpenRouter] key={_key_fingerprint(OPENROUTER_API_KEY)} "
    f"model={OPENROUTER_MODEL} vision={OPENROUTER_VISION_MODEL}"
)


def require_llm_client() -> OpenAI:
    """Return the OpenRouter client or raise a clear configuration error."""
    if deepseek_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenRouter API key is not configured. "
                "Add OPENROUTER_API_KEY=sk-or-... to .env and restart the backend."
            ),
        )
    return deepseek_client


GRAPH_USER_NODE = os.getenv("GRAPH_USER_NODE", "Affan Syed")

class PersonaTrainRequest(BaseModel):
    name: str
    social_urls: List[str] = []

class TextTrainRequest(BaseModel):
    persona: str = "My Personal Twin"
    text: str
    source_name: str = "Manual text training"

class TraitAddRequest(BaseModel):
    trait: str

class FeedbackRequest(BaseModel):
    message_id: str
    rating: str
    user_message: str = ""
    assistant_reply: str = ""
    persona_id: str = ""
    persona_name: str = "My Personal Twin"

TRAITS_FILE = os.path.join(UPLOAD_DIR, "persona_traits.json")
FEEDBACK_FILE = os.path.join(UPLOAD_DIR, "persona_feedback.json")

def load_feedback_store() -> List[Dict[str, Any]]:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading feedback store: {e}")
    return []

def save_feedback_store(data: List[Dict[str, Any]]):
    tmp = FEEDBACK_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, FEEDBACK_FILE)
    except Exception as e:
        print(f"Error saving feedback store: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def get_feedback_guidance(persona: str) -> str:
    records = [
        x for x in load_feedback_store()
        if str(x.get("persona_name", "")).strip().lower() == persona.strip().lower()
    ]
    records = records[-20:]
    if not records:
        return ""

    liked = [x for x in records if x.get("rating") == "up" and x.get("assistant_reply")]
    disliked = [x for x in records if x.get("rating") == "down" and x.get("assistant_reply")]
    parts = []

    if liked:
        parts.append(
            "The user marked these recent responses as GOOD. Preserve useful qualities shown in them:\n"
            + "\n---\n".join(x["assistant_reply"][:1200] for x in liked[-5:])
        )

    if disliked:
        parts.append(
            "The user marked these recent responses as BAD. Avoid repeating their wording, style, or approach:\n"
            + "\n---\n".join(x["assistant_reply"][:1200] for x in disliked[-5:])
        )

    return "\n\nFEEDBACK LEARNING:\n" + "\n\n".join(parts)

def init_traits_store():
    if not os.path.exists(TRAITS_FILE):
        default_traits = {
            "Affan": [
                "Calm, direct, and candid speaking style",
                "Street-smart and conversational with smooth explanations in Hinglish",
                "Motorcycle enthusiast who loves weekend breakfast rides",
                "Passionate about football, fitness, and gym workouts",
                "Fond of nature, mountains, and watching sunsets",
                "Values authenticity, discipline, and strongly dislikes dishonesty",
                "Enthusiastic about cyber security and artificial intelligence"
            ],
            "My Personal Twin": [
                "Helpful, thoughtful, and authentic conversational style",
                "Curious learner with strong analytical problem solving",
                "Direct and articulate communicator"
            ]
        }
        try:
            with open(TRAITS_FILE, "w", encoding="utf-8") as f:
                json.dump(default_traits, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error initializing persona traits file: {e}")

init_traits_store()

def load_persona_traits_from_disk() -> Dict[str, List[str]]:
    if os.path.exists(TRAITS_FILE):
        try:
            with open(TRAITS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading persona traits: {e}")
    return {}

def save_persona_traits_to_disk(data: Dict[str, List[str]]):
    try:
        with open(TRAITS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving persona traits: {e}")

def get_all_traits_for_persona(persona: str) -> List[str]:
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    local_traits = disk_data.get(persona_clean, [])
    graph_traits = graph_db.get_persona_traits(persona_clean) if graph_db.driver else []
    
    seen = set()
    combined = []
    for t in local_traits + graph_traits:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            combined.append(t_clean)
    return combined

def store_traits_for_persona(persona: str, new_traits: List[str]) -> List[str]:
    persona_clean = persona.strip()
    if not persona_clean:
        return []
    disk_data = load_persona_traits_from_disk()
    existing = disk_data.get(persona_clean, [])
    seen = {t.lower() for t in existing}
    for t in new_traits:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            existing.append(t_clean)
            if graph_db.driver:
                graph_db.add_trait(persona_clean, t_clean)
    disk_data[persona_clean] = existing
    save_persona_traits_to_disk(disk_data)
    return existing

def remove_trait_from_persona(persona: str, trait_to_remove: str) -> List[str]:
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    existing = disk_data.get(persona_clean, [])
    updated = [t for t in existing if t.lower() != trait_to_remove.strip().lower()]
    disk_data[persona_clean] = updated
    save_persona_traits_to_disk(disk_data)
    if graph_db.driver:
        graph_db.delete_trait(persona_clean, trait_to_remove)
    return updated

def purge_persona_traits(persona: str):
    persona_clean = persona.strip()
    disk_data = load_persona_traits_from_disk()
    if persona_clean in disk_data:
        del disk_data[persona_clean]
        save_persona_traits_to_disk(disk_data)
    if graph_db.driver:
        graph_db.delete_persona_traits(persona_clean)

class TwitterScraper:
    """
    Uses twifork (a maintained, drop-in-compatible fork of twikit, installed
    as the `twikit` package name) to pull profile/tweet data via an
    authenticated cookie session. This does not bypass login or create
    sessions on its own -- it replays an existing authenticated cookie
    export, so a valid, non-expired twitter_cookies.json (see
    format_cookies.py) is required.
    """

    def __init__(self, cookies_file: Optional[str] = "twitter_cookies.json"):
        if cookies_file and not os.path.isabs(cookies_file):
            cookies_file = os.path.join(BASE_DIR, cookies_file)
        self.cookies_file = cookies_file

    def _make_client(self) -> Client:
        # impersonate="chrome124" uses the optional curl_cffi backend
        # (pip install "twifork[impersonate]") to route requests through a
        # real browser TLS fingerprint, avoiding some 403s from X. Falls
        # back to the default httpx backend if that extra isn't installed.
        try:
            return Client("en-US", impersonate="chrome124")
        except Exception as e:
            print(f"[twitter] impersonate backend unavailable ({e}); falling back to default client")
            return Client("en-US")

    async def _authenticated_client(self) -> Client:
        if not self.cookies_file or not os.path.exists(self.cookies_file):
            raise RuntimeError(
                f"Cookie file '{self.cookies_file}' not found. Run format_cookies.py "
                "against a fresh raw_cookies.json export first."
            )
        client = self._make_client()
        client.load_cookies(self.cookies_file)
        if not await client.is_logged_in():
            raise RuntimeError(
                "Twitter cookies are expired or invalid. Re-export cookies from a "
                "logged-in browser session and re-run format_cookies.py."
            )
        return client

    async def get_user_profile(self, username: str, tweet_count: int = 20) -> Dict[str, Any]:
        try:
            client = await self._authenticated_client()
            clean_username = username.strip().replace("@", "").split("?")[0]
            if not clean_username:
                return {"status": "error", "message": "Empty username."}

            user = await client.get_user_by_screen_name(clean_username)
            if not user:
                return {"status": "error", "message": f"User @{clean_username} not found."}

            tweets = await client.get_user_tweets(user.id, "Tweets", count=min(max(tweet_count, 1), 40))

            scraped_content = [
                f"Profile Name: {user.name} (@{user.screen_name})",
                f"Bio: {user.description or 'No bio'}",
                f"Followers: {getattr(user, 'followers_count', 'N/A')} | Following: {getattr(user, 'following_count', 'N/A')}",
                "Recent Tweets:"
            ]
            for tweet in tweets:
                text = getattr(tweet, "full_text", None) or tweet.text
                created = getattr(tweet, "created_at", "")
                scraped_content.append(f"- [{created}] {text}")

            if len(scraped_content) == 4:  # no tweets appended beyond the header lines
                scraped_content.append("- (No public tweets found.)")

            return {"status": "success", "data": "\n".join(scraped_content)}
        except RuntimeError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            print(f"[twitter] get_user_profile error for @{username}: {e}")
            return {"status": "error", "message": f"Failed to scrape @{username}: {e}"}

    async def search_tweets(self, query: str, limit: int = 10) -> List[str]:
        try:
            client = await self._authenticated_client()
            results = await client.search_tweet(query, "Latest", count=min(max(limit, 1), 20))
            out = []
            for tweet in results:
                text = getattr(tweet, "full_text", None) or tweet.text
                author = getattr(tweet.user, "screen_name", "unknown") if getattr(tweet, "user", None) else "unknown"
                out.append(f"@{author}: {text}")
            return out
        except RuntimeError as e:
            print(f"[twitter] search_tweets auth error: {e}")
            return []
        except Exception as e:
            print(f"[twitter] search_tweets error for query '{query}': {e}")
            return []

twitter_client = TwitterScraper()

def extract_text_from_file(file_path: str, filename: str) -> str:
    text_content = ""
    ext = filename.lower().split(".")[-1]
    try:
        if ext == "pdf":
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
    return text_content.strip()

async def scrape_github_profile(url: str) -> str:
    try:
        clean_url = url.rstrip("/")
        parts = clean_url.split("/")
        username = parts[-1] if "github.com" in parts else parts[-2] if len(parts) > 1 else ""
        if not username:
            return "Invalid GitHub URL"
            
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
            profile_resp = await client.get(f"https://api.github.com/users/{username}", headers=headers)
            bio = "No bio found"
            fullname = username
            public_repos_count = 0
            if profile_resp.status_code == 200:
                p_data = profile_resp.json()
                bio = p_data.get("bio") or "No bio found"
                fullname = p_data.get("name") or username
                public_repos_count = p_data.get("public_repos", 0)

            repos_resp = await client.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated", headers=headers)
            projects = []
            repo_names = []
            if repos_resp.status_code == 200:
                repos_data = repos_resp.json()
                for repo in repos_data:
                    r_name = repo.get("name", "Unknown")
                    repo_names.append(r_name)
                    r_desc = repo.get("description") or "No desc"
                    if len(r_desc) > 80:
                        r_desc = r_desc[:77] + "..."
                    r_lang = repo.get("language") or "Mixed"
                    projects.append(f"- {r_name} ({r_lang}): {r_desc}")
            
            projects_text = "\n".join(projects) if projects else "No public repositories found."
            repo_list_summary = f"Key Repository Names: {', '.join(repo_names)}\n\n"
            
            return f"GitHub User: {fullname} (@{username})\nTotal Public Repositories: {public_repos_count}\nBio: {bio}\n\n{repo_list_summary}Repositories:\n{projects_text}"
            
    except Exception as e:
        print(f"GitHub API scraping error: {e}")
        return "Could not retrieve comprehensive GitHub profile details."

async def scrape_linkedin_profile(url: str) -> str:
    return "LinkedIn profile public scraping requires authentication. Please upload your resume or profile text directly."

async def scrape_twitter_profile(url: str) -> str:
    handle = url.strip().split("/")[-1].replace("@", "").split("?")[0]
    profile_data = await twitter_client.get_user_profile(handle)
    if profile_data.get("status") == "success":
        return f"Twitter/X Profile Data for @{handle}:\n{profile_data.get('data')}"
    return f"Could not retrieve tweets for @{handle} via twifork: {profile_data.get('message')}. Please verify cookies.json setup."

def is_instagram_url(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower().split(":")[0]
        return host == "instagram.com" or host.endswith(".instagram.com")
    except Exception:
        return False


async def scrape_instagram_public(url: str) -> Dict[str, Any]:
    """
    Retrieve Instagram page information.

    Instagram frequently serves a login/interstitial page even for public
    profiles. This function therefore supports two modes:

    1. Public HTML/oEmbed retrieval (no credentials).
    2. Optional authenticated browser-session retrieval using the
       INSTAGRAM_SESSIONID environment variable.

    The session cookie is never returned to the frontend and should never be
    committed to GitHub.
    """
    normalized_url = url.strip()
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = "https://" + normalized_url

    parsed_input = urlparse(normalized_url)
    path = parsed_input.path.rstrip("/") or "/"
    normalized_url = f"https://www.instagram.com{path}/"

    # Optional browser session. This is the important fix for the error shown
    # in the UI: Instagram can return HTTP 200 while still giving an anonymous
    # login/interstitial page.
    sessionid = os.getenv("INSTAGRAM_SESSIONID", "").strip()
    csrftoken = os.getenv("INSTAGRAM_CSRFTOKEN", "").strip()
    ds_user_id = os.getenv("INSTAGRAM_DS_USER_ID", "").strip()

    cookies = {}
    if sessionid:
        cookies["sessionid"] = sessionid
    if csrftoken:
        cookies["csrftoken"] = csrftoken
    if ds_user_id:
        cookies["ds_user_id"] = ds_user_id

    headers = {
        "User-Agent": os.getenv(
            "INSTAGRAM_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36",
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.instagram.com/",
        "Upgrade-Insecure-Requests": "1",
    }

    async def parse_html(html: str, final_url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")

        def meta(*names: str) -> str:
            for name in names:
                tag = soup.find("meta", attrs={"property": name})
                if not tag:
                    tag = soup.find("meta", attrs={"name": name})
                if tag and tag.get("content"):
                    return tag.get("content", "").strip()
            return ""

        title = meta("og:title", "twitter:title")
        description = meta("og:description", "twitter:description")
        image = meta("og:image", "twitter:image")
        published = meta("article:published_time")
        canonical = meta("og:url") or final_url
        author = ""

        # JSON-LD is often present on public Instagram pages.
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                raw = script.string or script.get_text()
                if not raw.strip():
                    continue
                data = json.loads(raw)
                objects = data if isinstance(data, list) else [data]
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue

                    a = obj.get("author")
                    if isinstance(a, dict):
                        author = author or str(
                            a.get("name") or a.get("alternateName") or ""
                        ).strip()
                    elif isinstance(a, str):
                        author = author or a.strip()

                    description = description or str(
                        obj.get("description") or ""
                    ).strip()

                    if not published:
                        published = str(
                            obj.get("datePublished")
                            or obj.get("uploadDate")
                            or ""
                        ).strip()

                    if not image:
                        image_value = obj.get("image")
                        if isinstance(image_value, str):
                            image = image_value
                        elif isinstance(image_value, list) and image_value:
                            image = str(image_value[0])
                        elif isinstance(image_value, dict):
                            image = str(image_value.get("url") or "")

            except Exception:
                continue

        # Extract visible text after removing scripts/styles.
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()

        visible_text = soup.get_text(separator=" ", strip=True)

        # Also inspect the document title as a fallback.
        html_title = ""
        if soup.title and soup.title.string:
            html_title = soup.title.string.strip()

        if not title and html_title and html_title.lower() != "instagram":
            title = html_title

        # Do not reject a page merely because Instagram also included a login
        # marker. If useful metadata is present, it is still useful.
        useful_metadata = bool(title or description or author or image)

        if not useful_metadata:
            return {
                "status": "error",
                "message": (
                    "Instagram returned a login/interstitial page or did not "
                    "expose readable public metadata."
                ),
            }

        path_lower = urlparse(final_url).path.lower()
        if "/reel/" in path_lower or "/reels/" in path_lower:
            page_type = "Reel"
        elif "/p/" in path_lower:
            page_type = "Post"
        elif "/tv/" in path_lower:
            page_type = "Video"
        else:
            page_type = "Profile/Page"

        parts = [
            "ACTIVE INSTAGRAM SOURCE",
            f"Page type: {page_type}",
            f"Source URL: {canonical}",
        ]

        if author:
            parts.append(f"Author: {author}")
        if title:
            parts.append(f"Title: {title}")
        if description:
            parts.append(f"Caption/Description: {description}")
        if published:
            parts.append(f"Published: {published}")
        if image:
            parts.append(f"Image URL: {image}")
        if visible_text:
            parts.append("Public page text:\n" + visible_text[:12000])

        return {
            "status": "success",
            "url": normalized_url,
            "final_url": final_url,
            "page_type": page_type,
            "author": author,
            "title": title,
            "caption": description,
            "published": published,
            "image": image,
            "text": "\n".join(parts),
            "authenticated": bool(sessionid),
        }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers=headers,
            cookies=cookies,
        ) as client:

            # First attempt: the normal Instagram page. If a valid browser
            # session was supplied, this request is authenticated.
            response = await client.get(normalized_url)

            if response.status_code == 200:
                parsed = await parse_html(response.text, str(response.url))
                if parsed.get("status") == "success":
                    return parsed

            # Second attempt: Instagram oEmbed for individual public posts,
            # reels and videos. This does not reliably support profile URLs.
            try:
                oembed_url = (
                    "https://www.instagram.com/oembed/?url="
                    + quote(normalized_url, safe="")
                )
                oembed_resp = await client.get(
                    oembed_url,
                    headers={**headers, "Accept": "application/json"},
                )

                if oembed_resp.status_code == 200:
                    data = oembed_resp.json()
                    author = str(data.get("author_name") or "").strip()
                    title = str(data.get("title") or "").strip()
                    thumbnail = str(data.get("thumbnail_url") or "").strip()

                    if author or title or thumbnail:
                        text_parts = [
                            "ACTIVE INSTAGRAM PUBLIC SOURCE",
                            "Page type: Public Instagram content",
                            f"Source URL: {normalized_url}",
                        ]
                        if author:
                            text_parts.append(f"Author: {author}")
                        if title:
                            text_parts.append(f"Title/Caption: {title}")
                        if thumbnail:
                            text_parts.append(f"Thumbnail URL: {thumbnail}")

                        return {
                            "status": "success",
                            "url": normalized_url,
                            "final_url": normalized_url,
                            "page_type": "Public Instagram content",
                            "author": author,
                            "title": title,
                            "caption": title,
                            "published": "",
                            "image": thumbnail,
                            "text": "\n".join(text_parts),
                            "authenticated": bool(sessionid),
                        }

            except Exception as oembed_err:
                print(f"[instagram] oEmbed fallback failed: {oembed_err}")

            if sessionid:
                message = (
                    f"Instagram still did not expose readable information "
                    f"(HTTP {response.status_code}). Your Instagram session "
                    f"may have expired or Instagram may be blocking automated "
                    f"requests."
                )
            else:
                message = (
                    f"Instagram returned HTTP {response.status_code} or a "
                    "login/interstitial page. Add INSTAGRAM_SESSIONID to the "
                    "backend .env and restart the server, then try again."
                )

            return {
                "status": "error",
                "message": message,
            }

    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Instagram request timed out.",
        }
    except httpx.RequestError as e:
        return {
            "status": "error",
            "message": f"Instagram request failed: {e}",
        }
    except Exception as e:
        print(f"[instagram] public/authenticated retrieval error: {e}")
        return {
            "status": "error",
            "message": f"Could not retrieve Instagram information: {e}",
        }


async def scrape_url_content(url: str) -> str:
    """Retrieve useful text from a URL with special handling for major platforms."""
    url_lower = url.lower()

    if is_instagram_url(url):
        result = await scrape_instagram_public(url)
        if result.get("status") == "success":
            return result.get("text", "")
        return "INSTAGRAM_ERROR: " + result.get("message", "Instagram public retrieval failed.")

    if "github.com" in url_lower:
        return await scrape_github_profile(url)
    if "linkedin.com" in url_lower:
        return await scrape_linkedin_profile(url)
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return await scrape_twitter_profile(url)

    # ------------------------------------------------------------------
    # WIKIPEDIA
    # Wikimedia can return 403 to automated clients. Use the Action API
    # first with both User-Agent and Api-User-Agent, then REST summary,
    # then a normal article request as a final fallback.
    # ------------------------------------------------------------------
    if "wikipedia.org/wiki/" in url_lower:
        try:
            from urllib.parse import urlparse, unquote, quote

            parsed = urlparse(url)
            marker = "/wiki/"
            if marker not in parsed.path:
                return "WIKIPEDIA_ERROR: Invalid Wikipedia URL."

            page_title = unquote(parsed.path.split(marker, 1)[1]).replace("_", " ").strip()
            if not page_title:
                return "WIKIPEDIA_ERROR: Could not determine the Wikipedia article name."

            encoded_title = quote(page_title.replace(" ", "_"), safe="")
            headers = {
                "User-Agent": "FACADE-PersonaTwin/1.2 (educational project; contact: facade@example.com) httpx",
                "Api-User-Agent": "FACADE-PersonaTwin/1.2 (educational project; contact: facade@example.com)",
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            }

            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                # 1) MediaWiki Action API - full plain-text extract
                action_params = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "titles": page_title,
                    "format": "json",
                    "formatversion": "2",
                }
                response = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params=action_params,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    pages = data.get("query", {}).get("pages", [])
                    if pages and not pages[0].get("missing"):
                        extract = (pages[0].get("extract") or "").strip()
                        if extract:
                            return (
                                f"WIKIPEDIA ARTICLE: {page_title}\n\n"
                                f"SOURCE: {url}\n\n{extract[:20000]}"
                            )

                # 2) REST summary API - shorter but useful fallback
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                summary_response = await client.get(summary_url, headers=headers)
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    extract = (summary_data.get("extract") or "").strip()
                    if extract:
                        return (
                            f"WIKIPEDIA ARTICLE: {page_title}\n\n"
                            f"SOURCE: {url}\n\n{extract[:12000]}"
                        )

                # 3) Direct page HTML as final fallback
                page_response = await client.get(url, headers={
                    **headers,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                })
                if page_response.status_code == 200:
                    soup = BeautifulSoup(page_response.text, "html.parser")
                    for element in soup(["script", "style", "noscript", "table"]):
                        element.extract()
                    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
                    if text:
                        return f"WIKIPEDIA ARTICLE: {page_title}\n\nSOURCE: {url}\n\n{text[:20000]}"

                # 4) Reader fallback for networks/IPs where Wikimedia blocks
                # server-side requests. This is only used after Wikimedia
                # itself rejects all three official endpoints.
                reader_url = "https://r.jina.ai/" + url
                reader_response = await client.get(
                    reader_url,
                    headers={"User-Agent": "FACADE-PersonaTwin/1.2"},
                )
                if reader_response.status_code == 200:
                    reader_text = re.sub(
                        r"\s+", " ", reader_response.text.strip()
                    ).strip()
                    if reader_text:
                        return (
                            f"WIKIPEDIA ARTICLE: {page_title}\n\n"
                            f"SOURCE: {url}\n\n{reader_text[:20000]}"
                        )

                statuses = f"Action API={response.status_code}, REST={summary_response.status_code}, page={page_response.status_code}, reader={reader_response.status_code}"
                return (
                    f"WIKIPEDIA_ERROR: Wikipedia could not be retrieved ({statuses}). "
                    "The article was blocked or unavailable from this machine."
                )

        except Exception as e:
            print(f"Wikipedia scraping error: {e}")
            return f"WIKIPEDIA_ERROR: {e}"

    # ------------------------------------------------------------------
    # GENERIC PUBLIC WEBPAGE
    # ------------------------------------------------------------------
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code != 200:
            return f"Could not retrieve content from URL (Status code: {response.status_code})"

        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return "The webpage did not contain readable text."
        return text[:12000]

    except Exception as e:
        print(f"Scraping error for {url}: {e}")
        return "Could not retrieve content from the URL due to a connection or rendering timeout."

async def scrape_photos_from_url(url: str) -> List[str]:
    img_urls = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.find_all('img'):
                    src = img.get('src')
                    if src and "http" in src:
                        if "s150x150" not in src and "emoji" not in src and "blank" not in src and "avatar" not in src:
                            img_urls.append(src)
    except Exception as e:
        print(f"Photo scraping error: {e}")
    return list(set(img_urls))[:15]

def extract_facts_and_traits_with_llm(text_content: str) -> dict:
    excerpt = text_content[:3500]
    extraction_prompt = (
        "Analyze the following text about a person and extract two distinct categories:\n"
        "1. 'facts': Factual attributes, relationships, skills, education, projects, or experiences.\n"
        "2. 'traits': Personality traits, communication style, tone of voice, behavioral habits, "
        "temperament, core values, mannerisms, and attitudes.\n\n"
        "Return ONLY a JSON object (no markdown, no commentary) with this exact structure:\n"
        "{\n"
        '  "facts": [\n'
        '    {"relation": "LIKES", "target": "Football"},\n'
        '    {"relation": "WORKS_WITH", "target": "Python"}\n'
        "  ],\n"
        '  "traits": [\n'
        '    "Calm and direct communication style",\n'
        '    "Passionate motorcycle enthusiast",\n'
        '    "High value on authenticity and honesty"\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- In 'facts', 'relation' must be an UPPER_SNAKE_CASE verb phrase (e.g. LIKES, WORKS_WITH, STUDIED_AT, DEVELOPED, PET, AIMS_FOR)\n"
        "- In 'facts', 'target' must be a short noun phrase\n"
        "- In 'traits', each trait must be a concise, expressive string describing their personality, tone, vibe, or behavioral attitude (extract 4 to 8 distinct traits)\n"
        "- If nothing extractable is found for either, use an empty list []\n\n"
        f"TEXT:\n{excerpt}"
    )
    try:
        client = require_llm_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            extra_body={"models": OPENROUTER_FALLBACK_MODELS},
            messages=[
                {"role": "system", "content": "You are a precise information and personality extraction engine. Output valid JSON only."},
                {"role": "user", "content": extraction_prompt}
            ],
            stream=False,
            timeout=15
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.endswith("json"):
                raw = raw[:-4]
        data = json.loads(raw)
        
        # Handle cases where LLM directly returns a list of facts
        if isinstance(data, list):
            data = {"facts": data, "traits": []}
        elif not isinstance(data, dict):
            return {"facts": [], "traits": []}
            
        raw_facts = data.get("facts", [])
        cleaned_facts = []
        if isinstance(raw_facts, list):
            for f in raw_facts:
                if isinstance(f, dict):
                    relation = str(f.get("relation", "")).strip().upper().replace(" ", "_")
                    target = str(f.get("target", "")).strip()
                    if relation and target and relation.replace("_", "").isalnum():
                        cleaned_facts.append({"relation": relation, "target": target})
                        
        raw_traits = data.get("traits", [])
        cleaned_traits = []
        if isinstance(raw_traits, list):
            for t in raw_traits:
                if isinstance(t, str):
                    t_clean = t.strip()
                    if t_clean:
                        cleaned_traits.append(t_clean)
                elif isinstance(t, dict):
                    t_str = str(t.get("trait") or t.get("name") or "").strip()
                    if t_str:
                        cleaned_traits.append(t_str)
                        
        return {"facts": cleaned_facts, "traits": cleaned_traits}
    except Exception as e:
        print(f"LLM fact & trait extraction failed: {e}")
        return {"facts": [], "traits": []}

def extract_facts_with_llm(text_content: str) -> list[dict]:
    # Backward-compatible wrapper
    return extract_facts_and_traits_with_llm(text_content).get("facts", [])

def save_profile_text_to_neo4j(filename: str, text_content: str, persona: str) -> dict:
    clean_persona = persona.strip() if persona else "My Personal Twin"
    extracted = extract_facts_and_traits_with_llm(text_content)
    facts = extracted.get("facts", [])
    traits = extracted.get("traits", [])
    
    # Store traits locally in persistent storage
    store_traits_for_persona(clean_persona, traits)
    
    if graph_db.driver:
        try:
            with graph_db.driver.session() as session:
                session.run(
                    "MERGE (u:Entity {name: $uname}) SET u.type = 'User'",
                    uname=clean_persona
                )
                for fact in facts:
                    graph_db.add_fact(clean_persona, fact["relation"], fact["target"])
                for trait in traits:
                    graph_db.add_trait(clean_persona, trait)
                session.run(
                    """
                    MATCH (u:Entity {name: $uname})
                    MERGE (d:Document {name: $fname})
                    MERGE (u)-[:UPLOADED]->(d)
                    """,
                    uname=clean_persona, fname=filename
                )
        except Exception as e:
            print(f"Neo4j sync warning: {e}")
    else:
        print("Neo4j driver not connected; saved traits to local JSON store.")
        
    return {
        "facts_added": len(facts),
        "facts": facts,
        "traits_added": len(traits),
        "traits": traits
    }



class ScrapeURLRequest(BaseModel):
    url: str
    persona: str = "My Personal Twin"
    persist: bool = True


@app.post("/scrape-url")
async def scrape_url_endpoint(request: ScrapeURLRequest):
    url = request.url.strip()
    persona = request.persona.strip() or "My Personal Twin"

    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(status_code=400, detail="Please enter a valid http:// or https:// URL")

    try:
        # Instagram is conversation context only. Never put it into the
        # persona's permanent Qdrant/Neo4j memory from this endpoint.
        if is_instagram_url(url):
            result = await scrape_instagram_public(url)
            if result.get("status") != "success":
                raise HTTPException(
                    status_code=502,
                    detail=result.get("message", "Instagram public retrieval failed.")
                )

            context = (result.get("text") or "").strip()
            if not context:
                raise HTTPException(status_code=502, detail="Instagram did not expose usable public information.")

            return {
                "status": "success",
                "persona": persona,
                "url": url,
                "final_url": result.get("final_url", url),
                "source_type": "instagram",
                "page_type": result.get("page_type", ""),
                "author": result.get("author", ""),
                "title": result.get("title", ""),
                "caption": result.get("caption", ""),
                "published": result.get("published", ""),
                "image": result.get("image", ""),
                "persisted": False,
                "chunks_added": 0,
                "images_found": 1 if result.get("image") else 0,
                "facts_added": 0,
                "traits_added": 0,
                "preview": context[:1500],
                "context": context[:15000],
            }

        text = await scrape_url_content(url)
        photos = await scrape_photos_from_url(url)

        if (
            not text.strip()
            or text.startswith("WIKIPEDIA_ERROR:")
            or text.startswith("INSTAGRAM_ERROR:")
            or text.lower().startswith("could not retrieve")
            or text.lower().startswith("wikipedia request failed")
        ):
            clean_error = text.split(":", 1)[1].strip() if ":" in text else text
            raise HTTPException(status_code=502, detail=clean_error or "Could not retrieve content from the URL")

        corpus = f"Source URL: {url}\n\n{text[:12000]}"
        if photos:
            corpus += "\n\nImages discovered on page:\n" + "\n".join(photos)

        should_persist = bool(request.persist)
        chunks_added = 0
        graph_result = {"facts_added": 0, "facts": [], "traits_added": 0, "traits": []}

        if should_persist:
            chunks_added = process_and_store_document(None, persona=persona, direct_text=corpus)
            graph_result = save_profile_text_to_neo4j(
                f"url_scrape_{url.split('/')[-1][:80] or 'page'}",
                corpus,
                persona
            )

        return {
            "status": "success",
            "persona": persona,
            "url": url,
            "source_type": "web",
            "persisted": should_persist,
            "chunks_added": chunks_added,
            "images_found": len(photos),
            "facts_added": graph_result.get("facts_added", 0),
            "traits_added": graph_result.get("traits_added", 0),
            "preview": text[:1200],
            "context": corpus[:15000],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"URL scrape endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape-photo")
async def scrape_photo_endpoint(file: UploadFile = File(...), persona: str = Form("My Personal Twin")):
    clean_persona = persona.strip() or "My Personal Twin"
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG, WEBP, or GIF image")
    try:
        raw = await file.read()
        if len(raw) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 12 MB.")
        mime = file.content_type or "image/jpeg"
        data_url = "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
        client = require_llm_client()
        response = client.chat.completions.create(
            model=OPENROUTER_VISION_MODEL,
            extra_body={"models": OPENROUTER_VISION_FALLBACK_MODELS},
            messages=[
                {"role":"system","content":"Extract useful personal/profile information from images. Return plain text only. Include visible text, names, roles, skills, projects, interests, and other relevant facts. Do not invent details."},
                {"role":"user","content":[
                    {"type":"text","text":"Analyze this image for information that should be added to the persona's personal knowledge base. Transcribe important visible text and summarize useful factual information."},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]}
            ], stream=False, timeout=60
        )
        extracted = (response.choices[0].message.content or "").strip()
        if not extracted:
            raise HTTPException(status_code=502, detail="The vision model returned no information from the image")
        corpus = f"Source image: {file.filename}\n\n{extracted}"
        chunks_added = process_and_store_document(None, persona=clean_persona, direct_text=corpus)
        graph_result = save_profile_text_to_neo4j(f"image_scrape_{file.filename}", corpus, clean_persona)
        return {"status":"success","persona":clean_persona,"filename":file.filename,
                "chunks_added":chunks_added,"facts_added":graph_result.get("facts_added",0),
                "traits_added":graph_result.get("traits_added",0),"text":extracted[:5000]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Photo scrape endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/persona/{name}/traits")
async def get_persona_traits_endpoint(name: str):
    traits = get_all_traits_for_persona(name)
    return {"persona": name, "traits": traits}

@app.post("/persona/{name}/traits")
async def add_persona_trait_endpoint(name: str, body: TraitAddRequest):
    clean_trait = body.trait.strip()
    if not clean_trait:
        raise HTTPException(status_code=400, detail="Trait cannot be empty")
    traits = store_traits_for_persona(name, [clean_trait])
    return {"status": "success", "persona": name, "traits": traits}

@app.delete("/persona/{name}/traits/{trait}")
async def delete_persona_trait_endpoint(name: str, trait: str):
    traits = remove_trait_from_persona(name, trait)
    return {"status": "success", "persona": name, "traits": traits}

@app.delete("/persona/{name}")
async def delete_persona_data(name: str):
    try:
        clean_name = name.strip()
        try:
            db_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="persona",
                            match=MatchValue(value=clean_name)
                        )
                    ]
                )
            )
        except Exception as q_err:
            print(f"Qdrant deletion warning: {q_err}")
        if graph_db.driver:
            with graph_db.driver.session() as session:
                session.run(
                    "MATCH (u:Entity {name: $uname}) DETACH DELETE u",
                    uname=clean_name
                )
        purge_persona_traits(clean_name)
        return {"status": "success", "message": f"Successfully purged persona '{clean_name}' from databases."}
    except Exception as e:
        print(f"Error deleting persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), persona: str = Form("My Personal Twin")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        text_data = extract_text_from_file(file_path, file.filename)
        if not text_data:
            raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty")
        chunk_count = process_and_store_document(file_path, persona=persona)
        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="Could not index text chunks into vector database")
        graph_result = {"facts_added": 0, "facts": []}
        graph_error = None
        try:
            graph_result = save_profile_text_to_neo4j(file.filename, text_data, persona=persona)
        except Exception as e:
            graph_error = str(e)
            print(f"Neo4j sync error: {graph_error}")
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_stored": chunk_count,
            "graph_facts_added": graph_result.get("facts_added", 0),
            "graph_facts": graph_result.get("facts", []),
            "traits_added": graph_result.get("traits_added", 0),
            "traits": graph_result.get("traits", []),
            "graph_error": graph_error,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-text")
async def train_text(request: TextTrainRequest):
    """Permanently add manually entered text to a persona's knowledge."""
    try:
        persona = request.persona.strip() or "My Personal Twin"
        text = request.text.strip()
        source_name = request.source_name.strip() or "Manual text training"

        if not text:
            raise HTTPException(status_code=400, detail="Training text cannot be empty.")
        if len(text) > 50000:
            raise HTTPException(status_code=400, detail="Training text is too long. Keep it under 50,000 characters.")

        training_text = (
            f"Manual training for persona: {persona}\n"
            f"Source: {source_name}\n\n"
            f"{text}"
        )

        chunks_added = process_and_store_document(
            None, persona=persona, direct_text=training_text
        )
        if chunks_added <= 0:
            raise HTTPException(status_code=500, detail="No knowledge chunks were indexed.")

        graph_result = {"facts_added": 0, "facts": [], "traits_added": 0, "traits": []}
        try:
            graph_result = save_profile_text_to_neo4j(
                f"{persona}_manual_text_{source_name[:60]}",
                training_text,
                persona=persona
            )
        except Exception as graph_err:
            print(f"Graph sync warning during text training: {graph_err}")

        return {
            "status": "success",
            "message": f"Successfully learned the text for persona '{persona}'.",
            "persona": persona,
            "source_name": source_name,
            "chunks_added": chunks_added,
            "facts_added": graph_result.get("facts_added", 0),
            "facts": graph_result.get("facts", []),
            "traits_added": graph_result.get("traits_added", 0),
            "traits": graph_result.get("traits", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Text training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-persona")
async def train_persona(request: PersonaTrainRequest):
    try:
        scraped_summaries = []
        for url in request.social_urls:
            cleaned_url = url.strip()
            if not cleaned_url.startswith("http://") and not cleaned_url.startswith("https://"):
                cleaned_url = "https://" + cleaned_url
            try:
                scraped_text = await scrape_url_content(cleaned_url)
                if "Could not retrieve" in scraped_text or not scraped_text.strip():
                    scraped_text = f"Profile handle or page for {request.name} at {cleaned_url}"
                scraped_summaries.append(f"Source URL ({cleaned_url}): {scraped_text[:3500]}")
            except Exception as scrape_err:
                scraped_summaries.append(f"Profile/Link: {cleaned_url}")
                
        extra_context = "\n\n".join(scraped_summaries) if scraped_summaries else f"Persona entity: {request.name}"
        training_text = f"Persona Profile Name: {request.name}.\n\nContent:\n{extra_context}"
        
        process_and_store_document(None, persona=request.name, direct_text=training_text)
        graph_result = {"facts_added": 0, "facts": [], "traits_added": 0, "traits": []}
        try:
            graph_result = save_profile_text_to_neo4j(f"{request.name}_profile", training_text, persona=request.name)
        except Exception as e:
            print(f"Graph sync error during persona training: {e}")
            
        return {
            "status": "success",
            "message": f"Successfully processed and trained persona '{request.name}'.",
            "facts_added": graph_result.get("facts_added", 0),
            "facts": graph_result.get("facts", []),
            "traits_added": graph_result.get("traits_added", 0),
            "traits": graph_result.get("traits", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    try:
        rating = request.rating.strip().lower()
        if rating not in {"up", "down"}:
            raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")

        persona_name = request.persona_name.strip() or "My Personal Twin"
        record = {
            "message_id": request.message_id.strip(),
            "rating": rating,
            "user_message": request.user_message[:5000],
            "assistant_reply": request.assistant_reply[:10000],
            "persona_id": request.persona_id[:200],
            "persona_name": persona_name,
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }

        records = load_feedback_store()
        # Replace an existing rating for the same message instead of creating duplicates.
        records = [r for r in records if not (r.get("message_id") == record["message_id"] and r.get("persona_name") == persona_name)]
        records.append(record)
        save_feedback_store(records)

        return {
            "status": "success",
            "message": "Feedback saved successfully",
            "persona": persona_name,
            "rating": rating,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Feedback save error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {e}")


@app.get("/api/feedback/{persona_name}")
async def get_feedback(persona_name: str):
    try:
        clean_name = persona_name.strip()
        records = [x for x in load_feedback_store() if str(x.get("persona_name", "")).strip().lower() == clean_name.lower()]
        return {"persona": clean_name, "feedback": records[-100:], "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_persona(
    message: str = Form(...),
    persona: str = Form("My Personal Twin"),
    history: str = Form(None),
    file: Optional[UploadFile] = File(None),
    url_context: str = Form(None),
    url_mode: str = Form(None),
):
    try:
        clean_persona = persona.strip() or "My Personal Twin"
        actual_message = message
        instagram_mode = str(url_mode or "").strip().lower() == "instagram"

        # "__UNAVAILABLE__" was used by an older frontend as a scrape-failure
        # marker. It is NOT an active source and must never poison later chat.
        if str(url_context or "").strip() == "__UNAVAILABLE__":
            url_context = None
            url_mode = None
            instagram_mode = False

        instagram_context_available = (
            instagram_mode
            and bool(url_context)
            and url_context.strip()
        )

        # Backend-side fallback: if a caller sends a raw Instagram URL directly
        # without using the frontend pre-scrape flow, retrieve it here too.
        raw_url_match = re.search(r"https?://[^\s<>\[\]\"']+", message or "", re.I)
        if raw_url_match and not instagram_mode:
            raw_target = raw_url_match.group(0).rstrip("),.!?")
            if is_instagram_url(raw_target):
                result = await scrape_instagram_public(raw_target)
                if result.get("status") != "success":
                    raise HTTPException(status_code=502, detail=result.get("message", "Instagram public retrieval failed."))
                url_context = result.get("text", "")[:15000]
                instagram_mode = True
                instagram_context_available = bool(url_context)
                actual_message = (
                    f"The user is asking about this Instagram URL: {raw_target}\n\n"
                    "Use the active Instagram source below as the primary evidence. "
                    "Do not claim access to anything not present in it."
                )

        # If the active source is Instagram but public retrieval failed, do
        # not let Qdrant/Neo4j/persona history answer the Instagram question.
        if instagram_mode and not instagram_context_available:
            return StreamingResponse(
                iter([
                    "I can't reliably answer questions about that Instagram page because Instagram did not expose usable public information to FACADE. I don't want to guess or use unrelated persona memory for it. Upload a screenshot of the page and I can analyze it directly. 😊"
                ]),
                media_type="text/plain"
            )

        # Backward-compatible [Scrape URL: ...] support. Instagram remains
        # temporary; other URLs retain the existing persistent behavior.
        url_match = re.search(r'\[Scrape URL:\s*(https?://[^\s\]]+)\]', message, re.I)
        if url_match:
            target_url = url_match.group(1)
            if is_instagram_url(target_url):
                result = await scrape_instagram_public(target_url)
                if result.get("status") != "success":
                    raise HTTPException(status_code=502, detail=result.get("message", "Instagram public retrieval failed."))
                url_context = result.get("text", "")[:15000]
                instagram_mode = True
                instagram_context_available = bool(url_context)
                actual_message = (
                    f"The user is asking about this Instagram URL: {target_url}\n\n"
                    "Use the active Instagram source below as the primary evidence. "
                    "Do not claim access to anything not present in it."
                )
            else:
                scraped_text = await scrape_url_content(target_url)
                scraped_photos = await scrape_photos_from_url(target_url)
                photos_text = "\n\nScraped Image URLs:\n" + "\n".join(scraped_photos) if scraped_photos else ""
                full_scraped_corpus = (
                    f"Here is the deep content, projects, and scraped code repositories from the profile {target_url}:\n\n"
                    f"{scraped_text}{photos_text}"
                )
                try:
                    process_and_store_document(None, persona=clean_persona, direct_text=full_scraped_corpus)
                    save_profile_text_to_neo4j(
                        f"url_scrape_{target_url.split('/')[-1]}",
                        full_scraped_corpus,
                        persona=clean_persona
                    )
                except Exception as persist_err:
                    print(f"Failed to auto-index scraped URL content: {persist_err}")
                actual_message = f"{full_scraped_corpus}\n\nPlease analyze this project data comprehensively for the user in your persona voice."

        file_attachment_context = ""
        image_base64_data = None
        if file and file.filename:
            file_bytes = await file.read()
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, safe_filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            if file.content_type and file.content_type.startswith("image/"):
                encoded_string = base64.b64encode(file_bytes).decode("utf-8")
                image_base64_data = f"data:{file.content_type};base64,{encoded_string}"
                file_attachment_context = f"\n\n[User attached an image named: {safe_filename}]"
            else:
                extracted_file_text = extract_text_from_file(file_path, safe_filename)
                if extracted_file_text:
                    file_attachment_context = f"\n\n[Attached File Content from {safe_filename}]:\n{extracted_file_text}"
                else:
                    file_attachment_context = f"\n\n[User attached a file named: {safe_filename}]"

        full_message_query = actual_message + file_attachment_context

        # Keep persona retrieval for normal chat. When Instagram is active,
        # it is explicitly labeled as background persona memory and is NOT
        # evidence for Instagram facts.
        query_vector = embed_text(full_message_query)
        search_result = db_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="persona",
                        match=MatchValue(value=clean_persona)
                    )
                ]
            )
        )
        vector_contexts = [hit.payload["text"] for hit in search_result.points if hit.payload]
        graph_facts = graph_db.get_related_facts(clean_persona)
        persona_traits = get_all_traits_for_persona(clean_persona)

        context_corpus = "\n".join(vector_contexts) if vector_contexts else "No document records found."
        graph_corpus = "\n".join(graph_facts) if graph_facts else "No graph records found."

        if persona_traits:
            traits_list_str = "\n".join([f"- {t}" for t in persona_traits])
            traits_section = (
                f"=== IDENTIFIED PERSONALITY & BEHAVIORAL TRAITS FOR '{clean_persona}' ===\n"
                "You possess the following core personality traits, behavioral patterns, and conversational tendencies:\n"
                f"{traits_list_str}\n\n"
                "MANDATORY BEHAVIORAL INSTRUCTION:\n"
                "You MUST actively and consistently embody these personality traits in your vocabulary, tone, perspective, humor, and sentence flow during the conversation. Do not simply list or quote your traits; speak naturally in exact alignment with them.\n\n"
            )
        else:
            traits_section = ""

        is_affan = clean_persona.lower() == "affan"
        if is_affan:
            style_instruction = (
                "NATURAL HINGLISH TEXTING & BILINGUAL STYLE FOR AFFAN:\n"
                "- Speak in a completely natural, casual, street-smart Indian conversational style (Hinglish), mixing Hindi words, phrases, and structures smoothly with English sentences within the same response.\n"
                "- Use natural flow like a real person chatting on WhatsApp.\n"
            )
        else:
            style_instruction = (
                f"DYNAMIC PERSONALITY & STYLE ADAPTATION FOR '{clean_persona}':\n"
                f"- Analyze the retrieved personal documents, writing style, tone, language preference, and vocabulary of '{clean_persona}' from the persona context.\n"
                f"- Adapt the response to '{clean_persona}'s voice and vibe without inventing factual information.\n"
            )

        feedback_guidance = get_feedback_guidance(clean_persona)

        instagram_section = ""
        if instagram_context_available:
            instagram_section = (
                "\n\n==================================================\n"
                "ACTIVE INSTAGRAM SOURCE — PRIMARY EVIDENCE\n"
                "==================================================\n"
                "The user is currently asking about an Instagram URL.\n"
                "For questions about that URL, the ACTIVE INSTAGRAM SOURCE below is the primary factual source.\n"
                "Persona memory, Qdrant results, Neo4j facts, old conversation claims, and assumptions are NOT evidence for Instagram-specific facts.\n"
                "If the requested detail is not in the active source, say that it is not available from the retrieved public page.\n"
                "NEVER invent posts, captions, dates, followers, likes, comments, stories, reels, or recent activity.\n\n"
                f"{url_context[:15000]}\n"
                "==================================================\n"
            )

        system_prompt = (
            f"CRITICAL INSTRUCTION: Your name and identity are strictly '{clean_persona}'. "
            f"You must embody this specific persona named '{clean_persona}' at all times. "
            f"If the user asks for your name, you must state that you are '{clean_persona}'.\n\n"
            f"{traits_section}"
            f"{style_instruction}\n"
            "STRICT FORMATTING PROHIBITIONS (ZERO FORMATTING TAGS):\n"
            "- NEVER USE ASTERISKS (*) OR HTML TAGS LIKE <b>, </b>, OR ANY OTHER MARKDOWN/HTML FORMATTING.\n"
            "- To highlight key terms or emphasize words, use UPPERCASE letters instead.\n"
            "- DO NOT use bullet points, list dashes (-), or numbered lists. Write in flowing conversational paragraphs.\n\n"
            f"PERSONA MEMORY / BACKGROUND CONTEXT:\n{context_corpus}\n\n"
            f"RELATED RELATIONSHIP OR ENTITY FACTS:\n{graph_corpus}\n"
            f"{instagram_section}\n"
            f"{feedback_guidance}\n\n"
            "FINAL SOURCE RULE:\n"
            "When an active Instagram source exists, answer Instagram-specific questions ONLY from that active source. "
            "Do not use persona memory or previous assistant messages to fill gaps. If information is unavailable, say so clearly.\n"
            f"Always respond as '{clean_persona}' and match the identified voice and traits.\n"
            "Use natural conversational emojis. Prefer emojis such as 🤞 😊 👍 ❤️ 👌 🙃 ☺️ 🫡 😜 🥱 😴 😪 🫠. "
            "Use at least one in every normal conversational reply, but do not spam them.\n"
            "Keep simple or casual replies short, usually 1-4 sentences. Give longer answers only when needed.\n"
            "Sound like a real person texting, not a formal AI assistant."
        )

        messages_payload = [{"role": "system", "content": system_prompt}]
        if history:
            try:
                parsed_history = json.loads(history)
                for turn in parsed_history:
                    role = turn.get("role")
                    content = turn.get("content")
                    if role in ["user", "assistant"] and content:
                        messages_payload.append({"role": role, "content": content})
            except Exception as hist_err:
                print(f"Error parsing history: {hist_err}")

        if image_base64_data:
            user_content_payload = [
                {"type": "text", "text": full_message_query},
                {"type": "image_url", "image_url": {"url": image_base64_data}},
            ]
        else:
            user_content_payload = full_message_query

        messages_payload.append({"role": "user", "content": user_content_payload})

        client = require_llm_client()
        llm_model = OPENROUTER_VISION_MODEL if image_base64_data else OPENROUTER_MODEL
        chat_fallbacks = (
            OPENROUTER_VISION_FALLBACK_MODELS if image_base64_data
            else OPENROUTER_FALLBACK_MODELS
        )
        response = client.chat.completions.create(
            model=llm_model,
            extra_body={"models": chat_fallbacks},
            messages=messages_payload,
            stream=True,
            timeout=90,
        )

        def generate_stream():
            try:
                emoji_pool = "🤞😊👍❤️👌🙃☺️🫡😜🥱😴😪🫠"
                full_output = []
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        clean_chunk = (
                            content_chunk
                            .replace("*", "")
                            .replace("<b>", "")
                            .replace("</b>", "")
                            .replace("<strong>", "")
                            .replace("</strong>", "")
                        )
                        full_output.append(clean_chunk)
                        yield clean_chunk

                final_text = "".join(full_output)
                if final_text.strip() and not any(e in final_text for e in emoji_pool):
                    yield " 😊"
            except Exception as stream_err:
                print(f"Streaming chunk error: {stream_err}")

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "Persona Twin backend running perfectly",
        "llm_configured": bool(OPENROUTER_API_KEY),
        "llm_provider": "OpenRouter",
        "api_key_fingerprint": _key_fingerprint(OPENROUTER_API_KEY),
        "model": OPENROUTER_MODEL,
        "fallback_models": OPENROUTER_FALLBACK_MODELS,
        "vision_model": OPENROUTER_VISION_MODEL,
        "neo4j_connected": bool(getattr(graph_db, "driver", None)),
    }

if __name__ == "__main__":
    import uvicorn
    # Render (and most Linux hosts) inject PORT and expect the app to bind
    # 0.0.0.0. "proactor" is a Windows-only asyncio loop, so it's only used
    # there; other platforms use uvicorn's default loop.
    run_kwargs = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", "8000")),
        "reload": False,
    }
    if sys.platform == "win32":
        run_kwargs["loop"] = "proactor"
    uvicorn.run("backend.main:app", **run_kwargs)