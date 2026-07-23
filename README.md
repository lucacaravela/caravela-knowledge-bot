# Caravela Knowledge Bot

Internal knowledge chatbot for **Caravela Capital**. The team logs in with
their Google accounts and asks questions like *"what healthcare SaaS
companies have we seen, what do they do, and what should I keep in mind when
talking to similar companies?"*. The bot answers by querying **Affinity CRM**
and **Google Drive** live at question time and synthesizing with the
**Claude API** (no database, no sync jobs).

**Stack:** Python 3.11+, a single Streamlit app (`app.py`), one module per
integration (`affinity.py`, `drive.py`), and the Claude agentic tool-use loop
in `agent.py` (model `claude-sonnet-4-6`, max 10 tool calls per question).

---

## 1. Local setup

```bash
git clone <this-repo>
cd <this-repo>

# Create and activate a virtualenv (Python 3.11+)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# ...then open .env and fill in every value (sections 2-4 below explain how
# to get each one).

# Run
streamlit run app.py
```

The app opens at http://localhost:8501. On first run it auto-generates
`.streamlit/secrets.toml` (needed by Streamlit's Google login) from your
`.env` values.

Run the tests with:

```bash
pytest
```

---

## 2. Affinity API key

1. Open Affinity in the browser and click your profile picture → **Settings**.
2. Go to the **API** section (you need **admin rights** on the Affinity
   account to see it — ask an admin if you don't).
3. Generate / copy the API key and put it in `.env` as `AFFINITY_API_KEY`.

The same key works for both Affinity APIs the bot uses: the v2 API
(companies, lists, fields — Bearer auth) and the v1 API (notes — Basic auth).

---

## 3. Google Cloud setup (Drive access via service account)

This gives the bot **read-only** access to the documents in Caravela's
shared drives. You have never used Google Cloud? Follow along exactly:

1. **Create a project**
   - Go to https://console.cloud.google.com and log in with your Google
     Workspace account.
   - Click the project dropdown (top-left) → **New project** → name it e.g.
     `caravela-knowledge-bot` → **Create**. Make sure it's selected.

2. **Enable the Drive API**
   - Menu ☰ → **APIs & Services** → **Library**.
   - Search for **Google Drive API** → open it → **Enable**.

3. **Create a service account**
   - Menu ☰ → **IAM & Admin** → **Service Accounts** → **Create service account**.
   - Name: `knowledge-bot-drive` → **Create and continue** → skip the
     optional role/access steps → **Done**.

4. **Download the JSON key**
   - Click the service account you just created → **Keys** tab →
     **Add key** → **Create new key** → type **JSON** → **Create**.
   - A `.json` file downloads. Keep it secret — it's a credential.
   - For local dev you can either point `GOOGLE_SERVICE_ACCOUNT_FILE` at the
     file, or paste the entire JSON **as one line** into
     `GOOGLE_SERVICE_ACCOUNT_JSON` (that's what Render will use).

5. **Share the Drive folders with the service account**
   - Open the service account details and copy its email — it looks like
     `knowledge-bot-drive@caravela-knowledge-bot.iam.gserviceaccount.com`.
   - In Google Drive, open the Caravela **shared drive** (or the specific
     folders with memos/decks) → **Manage members** / **Share** → add that
     email with **Viewer** (read-only) access.
   - The bot can only see what you share here. If Drive searches return
     nothing, this step is almost always the reason.

---

## 4. Google OAuth client (for the login screen)

This is separate from the service account — it's what lets teammates log in
with their own Google accounts.

1. Menu ☰ → **APIs & Services** → **OAuth consent screen**.
   - User type: **Internal** (only accounts in your Workspace can log in).
   - Fill in the app name (`Caravela Knowledge Bot`) and support emails →
     **Save**.
2. Menu ☰ → **APIs & Services** → **Credentials** → **Create credentials** →
   **OAuth client ID**.
   - Application type: **Web application**.
   - Name: `caravela-knowledge-bot`.
   - Under **Authorized redirect URIs**, add BOTH:
     - `http://localhost:8501/oauth2callback` (local dev)
     - `https://YOUR-APP.onrender.com/oauth2callback` (you'll know the exact
       URL after creating the Render service — come back and add it).
   - **Create**, then copy the **Client ID** and **Client secret** into
     `.env` as `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET`.
3. Set `AUTH_COOKIE_SECRET` to any long random string:
   `python -c "import secrets; print(secrets.token_hex(32))"`.
4. Set `AUTH_REDIRECT_URI` to the localhost URI above (on Render it will be
   the onrender.com one).
5. Set `ALLOWED_DOMAIN` to your email domain (e.g. `caravela.capital`) —
   only accounts on that domain get past the login.

---

## 5. Deploy to Render (free tier)

1. Push this repo to GitHub.
2. Go to https://render.com → **New** → **Blueprint** and connect the GitHub
   repo (the included `render.yaml` configures everything), **or** create a
   **Web Service** manually with:
   - Build command: `pip install -r requirements.txt`
   - Start command:
     `python auth_secrets.py && streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. In the Render dashboard, open the service → **Environment** and set:
   `ANTHROPIC_API_KEY`, `AFFINITY_API_KEY`, `GOOGLE_SERVICE_ACCOUNT_JSON`
   (the full JSON on one line), `ALLOWED_DOMAIN`, `GOOGLE_OAUTH_CLIENT_ID`,
   `GOOGLE_OAUTH_CLIENT_SECRET`, `AUTH_COOKIE_SECRET`, and
   `AUTH_REDIRECT_URI=https://YOUR-APP.onrender.com/oauth2callback`.
4. Add that same `https://YOUR-APP.onrender.com/oauth2callback` URL to the
   OAuth client's **Authorized redirect URIs** in Google Cloud (section 4).
5. Deploy. **Note:** the free tier spins down after ~15 minutes of
   inactivity — the first request after that takes about a minute to wake
   the service up. That's normal.

---

## 6. WhatsApp bot (optional)

A second entry point (`whatsapp_app.py`) serves the same bot on WhatsApp
via the Meta Cloud API. Pilot setup with Meta's free test number:

1. https://developers.facebook.com → Create App → use case **"Connect with
   customers through WhatsApp"** → create/select the business portfolio.
2. In the app: WhatsApp use case → **API Setup**. Copy the temporary
   **access token** and the **Phone number ID**, and register up to 5 team
   phone numbers as test recipients.
3. Deploy the `caravela-whatsapp-bot` service (already in `render.yaml`)
   and set: `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`,
   `WHATSAPP_VERIFY_TOKEN` (any string you invent), `WHATSAPP_APP_SECRET`
   (App settings → Basic), `ALLOWED_PHONES` (comma-separated, digits only,
   with country code), plus `ANTHROPIC_API_KEY` and `AFFINITY_API_KEY`.
4. In the Meta app, configure the webhook: callback URL
   `https://caravela-whatsapp-bot.onrender.com/webhook`, verify token =
   the same `WHATSAPP_VERIFY_TOKEN`, and subscribe to the **messages**
   field.
5. Message the test number from a registered phone.

Production later: real phone number + Meta business verification (CNPJ),
a permanent token via a system user, and the Render Starter plan so the
webhook never cold-starts. The temporary token expires every 24h — fine
for the pilot only.

## 7. Troubleshooting

**Affinity returns 401 Unauthorized**
The two Affinity APIs use different auth schemes: v2
(`https://api.affinity.co/v2`) wants `Authorization: Bearer <key>`, while v1
(`https://api.affinity.co`) wants HTTP Basic auth with an **empty username**
and the key as the password. A 401 usually means the wrong scheme hit the
wrong API, or the key was generated without admin rights. Regenerate the key
in Settings → API and make sure you copied it fully.

**Drive searches return nothing**
Almost always: the shared drive/folder was **not shared with the service
account email** (section 3, step 5). Share it with Viewer access and try
again. Also check that the Google Drive API is enabled in the same Cloud
project as the service account.

**OAuth error: `redirect_uri_mismatch`**
The `AUTH_REDIRECT_URI` env var and the **Authorized redirect URIs** in the
Google Cloud OAuth client must match *exactly* (scheme, host, port, and the
`/oauth2callback` path). Add the URI you're actually using — localhost for
dev, the onrender.com URL for production — and wait a minute for Google to
propagate the change.

**"Acesso negado" after logging in**
The account's email domain doesn't match `ALLOWED_DOMAIN`. Check the env var
(no `@`, just the domain, e.g. `caravela.capital`).

**Anthropic overloaded / rate limited**
The app retries with exponential backoff automatically. If it still fails,
wait a moment and try again, or check https://status.anthropic.com.
