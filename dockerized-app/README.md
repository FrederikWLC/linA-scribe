# Dockerized FastAPI + Svelte App

This folder contains a minimal Dockerized full-stack application:

- `src/backend`: FastAPI server
- `frontend`: Svelte app (Vite)
- `docker-compose.yml`: Starts both services together

## Run

From this directory:

```powershell
docker compose up --build
```

After the first build, source code edits are hot-reloaded in both services.

Open:

- Frontend: http://localhost:5173
- Backend health: http://localhost:8000/api/health

## Minimal Login

- Backend endpoint: `POST /api/auth/login`
- Request body: `{ "username": "alice", "password": "..." }`
- Behavior:
	- If password is wrong: login fails and user is not created.
	- If password is correct and username does not exist: user is created and logged in.
	- If password is correct and username exists: user logs in.
	- On success, response includes a session token.

Protected endpoints (require `Authorization: Bearer <token>`):

- `GET /api/protected/page`

Set universal password via environment variable:

- `UNIVERSAL_PASSWORD` (default: `linascribe`)
- `MASTER_PASSWORD` (default: `linascribe-master`)

Master-only endpoints (master password logins only):

- `GET /api/evaluation/page`
- `GET /api/evaluation/tuning/files`
- `GET /api/evaluation/evaluation/files`
- `GET /api/evaluation/ablation/files`
- `GET /api/evaluation/scribing/files`

Evaluation flow:

- Lists existing files from Supabase (signed URLs), split by process.
- Backend mounts the project-level `src` folder at `/workspace/src`.
- Configure host path with `PROJECT_SRC_HOST_PATH` in `.env` (default `../src`).

Frontend routes:

- `/login`: login page
- `/tool`: normal authenticated page
- `/evaluation`: master-only page

## Supabase Storage (Backend Only)

Supabase is integrated only in FastAPI (not in Svelte).

1. Create a Storage bucket in Supabase (for example `linascribe-files`).
2. Add these environment variables in this folder (for example via `.env`):
	- `SUPABASE_URL`
	- `SUPABASE_SERVICE_ROLE_KEY`
	- `SUPABASE_BUCKET`
3. Rebuild and run:

```powershell
docker compose up --build
```

Minimal backend endpoints:

- `POST /api/storage/upload?path=<bucket_path>`
- `GET /api/storage/download-url?path=<bucket_path>`

## Hot Reload Notes

- Backend (`FastAPI`): edits in `src/backend/app` trigger automatic reload.
- Frontend (`Svelte + Vite`): edits in `frontend/src` update in the browser.
- If you change dependencies (`requirements.txt` or `package.json`), rebuild images:

```powershell
docker compose up --build
```

## Stop

```powershell
docker compose down
```
