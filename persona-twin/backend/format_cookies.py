import json
import sys

RAW_COOKIES_PATH = "raw_cookies.json"
OUTPUT_PATH = "twitter_cookies.json"
# twikit/twifork's is_logged_in() check relies on these two being present and fresh.
REQUIRED_COOKIE_NAMES = {"auth_token", "ct0"}

try:
    # Read the raw JSON file (exported via a browser cookie-export extension,
    # e.g. "Cookie-Editor", while logged into x.com) instead of hardcoding a string.
    with open(RAW_COOKIES_PATH, "r", encoding="utf-8") as f:
        cookies_list = json.load(f)

    if not isinstance(cookies_list, list):
        print(f"Error: expected {RAW_COOKIES_PATH} to contain a JSON array of cookie objects.")
        sys.exit(1)

    # Convert it into the simple {name: value} format twikit/twifork requires
    twikit_cookies = {
        cookie["name"]: cookie["value"]
        for cookie in cookies_list
        if "name" in cookie and "value" in cookie
    }

    missing = REQUIRED_COOKIE_NAMES - twikit_cookies.keys()
    if missing:
        print(
            f"Warning: missing required cookie(s) {sorted(missing)}. "
            "The export may be incomplete or you weren't fully logged in; "
            "is_logged_in() will likely fail with this file."
        )

    # Save the correctly formatted cookies
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(twikit_cookies, f, indent=4)

    print(f"Success! {OUTPUT_PATH} has been created with {len(twikit_cookies)} cookies.")
except FileNotFoundError:
    print(f"Error: '{RAW_COOKIES_PATH}' not found. Export your cookies from a logged-in "
          "x.com browser session to that file first, then re-run this script.")
except Exception as e:
    print(f"Error parsing cookies: {e}")