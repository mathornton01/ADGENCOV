# ADGENCOV — Deploy Notes (for future sessions)

## Railway
- Project: `authentic-amazement` (ID `3a528d09-e1b4-4e67-ab2a-b364c7a32df4`)
- Service: `adgencov-api` (ID `901339c7-8ca7-42f1-8238-0f30c7655833`)
- LIVE URL: https://adgencov-api-production.up.railway.app  ← the real one
  - NOT `adgencov-production.up.railway.app` (Railway edge 404 fallback — a red herring)
  - `api.thorntonstatistical.com` custom domain currently fails TLS; ignore.

## Auth (persisted)
- Project token lives in `~/.railway/env` (chmod 600), which `~/.bashrc` sources.
- Agent non-interactive shells don't auto-source it, so before railway calls run:
  `set -a; . ~/.railway/env; set +a`
- `railway whoami` returns Unauthorized with a *project* token — that's expected, not a failure.

## Deploy (CLI-driven, Dockerfile builder)
- Service is unlinked in CLI; target it explicitly:
  `( cd <adgencov dir> && railway up --service adgencov-api --detach )`
- Poll: `railway deployment list | sed -n '2p'` until SUCCESS.

## Endpoint smoke test after deploy
- `GET /health` → `{"status":"ok",...}`
- `POST /translate/symbols` body `{"ids":["7157"],"organism":"human"}` → symbol `TP53`
- `GET /interactions?genes=TP53,MDM2&organism=human` → `direct` score + STRING partners
