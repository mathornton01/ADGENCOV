# Deploying the ADGENCOV service

The backend is a FastAPI app over the compiled C++ core. It is containerized
(`Dockerfile`) and deploys to any container host. Below is the Railway path,
which auto-deploys on every push to `main`.

## 1. One-time Railway setup (browser, ~2 min)

1. Sign in at https://railway.app (GitHub login recommended).
2. **New Project → Deploy from GitHub repo → `mathornton01/ADGENCOV`**.
   Authorize Railway to read the repo if prompted.
3. Railway detects `railway.json` + `Dockerfile` and builds automatically.
   First build takes ~5–8 min (it fetches Eigen + pybind11 and compiles the
   C++ core). Subsequent builds are cached and faster.
4. When the deploy is green, Railway assigns a URL like
   `https://adgencov-production.up.railway.app`. Hit `/health` to confirm.

After this, every `git push` to `main` triggers a new build+deploy — no CLI,
no manual step.

## 2. Point thorntonstatistical.com at it

In Railway: **Service → Settings → Networking → Custom Domain** → add
`api.thorntonstatistical.com` (a subdomain keeps the API separate from the
marketing site). Railway shows a `CNAME` target; add that record at your DNS
provider:

    CNAME  api  ->  <the target Railway shows>

TLS is provisioned automatically once DNS resolves. The app's CORS allow-list
already includes `https://thorntonstatistical.com` and `https://www.` — so a
page on the main site can call `https://api.thorntonstatistical.com` directly.

## 3. Config notes

- **Port**: the container binds `$PORT` (Railway injects it). Locally it
  defaults to 8000.
- **Health check**: `railway.json` points Railway at `/health`.
- **Portable binary**: the image builds with `-march=native` OFF so it runs on
  any CPU Railway schedules. The big parallel-grid speedup is in Python and is
  unaffected; only per-op SIMD width is traded for portability.
- **Long jobs**: analysis runs as an async job (submit → poll `GET /jobs/{id}`),
  so it never trips platform request timeouts regardless of dataset size.

## Local container test (optional, needs Docker)

    docker build -t adgencov .
    docker run --rm -p 8000:8000 adgencov
    curl localhost:8000/health
