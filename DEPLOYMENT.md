# ForestProof deployment guide

This repository deploys as two separate services: FastAPI on Render and Streamlit on Streamlit Community Cloud. The Streamlit frontend must point to the public Render backend with `FORESTPROOF_API_URL`.

## 1. Prepare and publish the GitHub repository

1. Review the project folder before making it public. Do not include `.env`, `.streamlit/secrets.toml`, personal field records, UAV imagery, `forestproof_outputs/`, or model files. The included `.gitignore` excludes common secrets and generated output.
2. Sign in to GitHub and create a repository named `ForestProof` (or another available name). Set visibility to **Public**. Do not initialize it with a README, license, or `.gitignore`, because those files are already in this project.
3. From the project root (the folder containing `app.py` and `main.py`), initialize and push the code:

   ```bash
   git init
   git add .
   git status --short
   git commit -m "Prepare ForestProof for public deployment"
   git branch -M main
   git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/ForestProof.git
   git push -u origin main
   ```

   Replace `<YOUR-GITHUB-USERNAME>` with your GitHub account or organization. GitHub may prompt you to authenticate in its own approved flow. Never put a personal access token in the remote URL or in a committed file.

4. Before confirming the push, inspect `git status` and the staged file list. If any secret or private dataset appears, stop, remove it from the staged changes, and rotate any credential that was already exposed.

## 2. Deploy the FastAPI backend on Render

1. In Render, create a **New Web Service** and connect the public `ForestProof` repository.
2. Use the project root as the service root directory. Select **Python 3**. Set the build command to:

   ```text
   pip install -r requirements.txt
   ```

3. Set the start command exactly to:

   ```text
   python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. Choose the Free plan for a hackathon demo if appropriate. If using the included `render.yaml`, Render can read the service name, build/start commands, and `/health` check from that Blueprint file.
5. Create/deploy the service and wait until Render reports it live. Copy the service's HTTPS URL, for example `https://forestproof-api.onrender.com`.
6. Verify these URLs in a browser:
   - `https://<your-render-service>.onrender.com/health` should return `{"status":"ok"}`.
   - `https://<your-render-service>.onrender.com/docs` should show the FastAPI OpenAPI page.

## 3. Deploy the Streamlit frontend

1. Open Streamlit Community Cloud and connect the same GitHub account if prompted.
2. Choose **Create app**, select the `ForestProof` repository and `main` branch, and set the main file path to `app.py`.
3. Open **Advanced settings** and set Python to 3.12 if the option is shown.
4. In the **Secrets** field, add the following TOML, replacing the value with the actual Render URL copied above:

   ```toml
   FORESTPROOF_API_URL = "https://<your-render-service>.onrender.com"
   ```

   Do not include `/health` at the end. Do not commit this as `.streamlit/secrets.toml`; set it only in the Streamlit app's settings.

5. Save/deploy and wait for the app to finish starting. Copy the generated `https://<your-app>.streamlit.app` URL.

## 4. Verify the connection and public app

1. Open the Render `/health` endpoint and confirm it reports `status: ok`.
2. Open the public Streamlit URL in a private/incognito browser window. The app should load the ICMBio Jamanxim boundary without showing the “backend is not reachable” error.
3. If it fails, open the Streamlit app settings and check that `FORESTPROOF_API_URL` is exactly the Render HTTPS base URL. Check Render deploy logs for startup errors, then redeploy if you change code.
4. Run the “Load verified Jamanxim historical demo case” workflow if the external INPE endpoint responds, then run a satellite analysis. External STAC, WFS, and imagery requests may be slow or temporarily unavailable; only report results that actually appear in the app.
5. Save the final URLs and perform the four-minute walkthrough in `DEMO_SCRIPT.md`.

## Operational limitations

The frontend calls FastAPI server-to-server using `FORESTPROOF_API_URL`, so browser CORS configuration is not required for this design. The API can be opened directly, but the user-facing interface is the Streamlit app.

Render's Free web services spin down after 15 minutes without inbound traffic and can take about a minute to start again. Their local filesystem is ephemeral; files written by ForestProof such as label records, history, evidence previews, and trained models can be lost when the service restarts, redeploys, or spins down. Treat the free deployment as a demo, not durable storage. See Render's [free service limitations](https://render.com/docs/free).

For local development, start the backend with `python -m uvicorn main:app --reload`, then start the frontend with `python -m streamlit run app.py`. The frontend defaults to `http://127.0.0.1:8000` when no secret or environment variable is set.
