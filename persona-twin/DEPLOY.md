# Deploying Persona Twin: Vercel (frontend) + Render (backend)

## 0. Do this first — rotate your credentials

Before you push anything to GitHub, rotate every credential below. This
isn't optional cleanup — `raw_cookies.json` and `twitter_cookies.json` were
committed to this project's git history at one point (later deleted, but
still recoverable from the history), and your `.env` file's own comment
notes the OpenRouter key should be replaced if it was ever pasted anywhere.
Treat all of these as already exposed:

- OpenRouter API key
- Qdrant API key
- Neo4j password
- X (Twitter) `auth_token` / `ct0` cookies
- Instagram `sessionid` / `csrftoken` / `ds_user_id`

The X and Instagram cookies are session credentials — anyone with them can
act as you on those platforms until the session is invalidated. Log out of
the sessions tied to those cookies (or change your password, which
invalidates sessions) and re-export fresh ones only if you still need the
scraping features.

When you push this project to GitHub, start a **fresh git history** (see
step 1) rather than pushing the existing `.git` folder, so the old secrets
in history don't end up on GitHub too.

## 1. Prepare the repo

This folder is already cleaned up for you (see "What was fixed" below), but
a couple of things are still worth doing explicitly:

```bash
cd persona-twin
rm -rf .git backend/.git   # remove any old git history/nested repos
git init
git add .
git commit -m "Initial commit"
```

Then push to a new GitHub repo. `.env`, `raw_cookies.json`,
`twitter_cookies.json`, and `venv/` are all in `.gitignore` — verify with
`git status` that none of them show up before you commit.

## 2. Deploy the backend on Render

1. New Web Service → connect your GitHub repo.
2. Render should detect `render.yaml` automatically (or set these manually):
   - **Root Directory**: `persona-twin` (only needed if your repo has the
     wrapper folder from the original zip — if `persona-twin` is the repo
     root itself, leave this blank)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Add the environment variables listed in `.env.example` with your
   **new, rotated** values. `QDRANT_URL`/`QDRANT_API_KEY` and
   `NEO4J_URI`/`NEO4J_PASSWORD` are required — the app raises an error on
   startup without them. `OPENROUTER_API_KEY` is optional at startup but
   required for chat to work.
4. Deploy, then open the service URL — `GET /` should return a JSON health
   check (`{"status": "Persona Twin backend running perfectly", ...}`).

## 3. Deploy the frontend on Vercel

The frontend is a single static `frontend/index.html` with no build step.

1. New Project → import the same GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Framework preset: **Other**. Leave build command empty — there's nothing
   to build.
4. Deploy.

## 4. Connect them

1. Open your deployed Vercel URL, click the settings/gear icon, and set
   **Backend URL** to your Render service URL (e.g.
   `https://persona-twin-backend.onrender.com`). This is saved in the
   browser's `localStorage`, so each visitor only needs to do this once per
   browser — but you'll want to set it yourself the first time you open it.
2. Back on Render, set `ALLOWED_ORIGINS` to your Vercel URL (e.g.
   `https://your-app.vercel.app`) and redeploy, so CORS isn't wide open to
   every site on the internet. It'll work with the default wildcard too,
   just less locked down.

## What was fixed in this copy

- **CORS**: the app previously combined `allow_origins=["*"]` with
  `allow_credentials=True`, which browsers reject for credentialed
  requests. Since the frontend never sends cookies, `allow_credentials` is
  now `False`, and `allow_origins` can be locked to your Vercel domain via
  the new `ALLOWED_ORIGINS` env var.
- **Port/host binding**: `backend/main.py`'s `__main__` block was hardcoded
  to `127.0.0.1:8000` and a Windows-only event loop. It now binds
  `0.0.0.0` and reads `$PORT`, and only uses the Windows loop on Windows.
  (Render actually runs the app via the `uvicorn` start command above, not
  this block, but it's now portable either way.)
- **Removed the Vercel full-stack setup**: the old root `vercel.json` and
  `api/index.py` ran the entire Python backend as a Vercel serverless
  function. Since the backend is moving to Render, those are removed —
  Vercel now just serves `frontend/index.html` as a static site.
- **Removed secrets and local artifacts from the deploy copy**:
  `backend/.env`, `raw_cookies.json`, `twitter_cookies.json`, `venv/`,
  `qdrant_data/`, `backend/data/`, and the nested `backend/.git` (a second,
  separate git repo living inside the main one — left in place, it would
  have caused your top-level git to store `backend/` as an empty submodule
  reference instead of tracking its files, so none of your backend code
  would actually reach GitHub).
- **Added `.env.example` and `render.yaml`** as templates — no real
  secrets in either.

## Things to know going in

- **Ephemeral uploads**: `UPLOAD_DIR` is `/tmp/facade_data`. On Render's
  free tier this is wiped on every restart/redeploy, so trait/feedback
  JSON files and uploaded documents living only there won't survive a
  redeploy. Qdrant and Neo4j data (your actual persona knowledge) does
  persist since those are external managed databases.
- **Cold starts**: Render's free tier spins down after inactivity; the
  first request after idling can take ~30-60s.
- **The Twitter/Instagram scraping features** require live, valid session
  cookies for a real logged-in account. If you don't plan to use them,
  just leave those env vars unset — everything else works fine without
  them.
