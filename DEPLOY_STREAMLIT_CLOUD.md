# Deploy Anomaly Dashboard to Streamlit Community Cloud

Steps to run the app on Streamlit Cloud with password protection and editable prompt (stored in Supabase).

---

## 1. What to keep in the repo (GitHub)

**Repo:** [github.com/SaiCIAI9/Anomaly-Detection](https://github.com/SaiCIAI9/Anomaly-Detection)

Push at least these so Streamlit Cloud can run the app:

| File / folder | Purpose |
|---------------|--------|
| `streamlit_anomaly_app.py` | Main app |
| `requirements.txt` | Dependencies (Streamlit Cloud looks for this by default) |
| `Anomalies_List.csv` | Data the app loads (same directory as the app) |
| `.streamlit/secrets.toml.example` | Example for local secrets (do **not** commit `secrets.toml`) |
| `DEPLOY_STREAMLIT_CLOUD.md` | This guide |

Optional: `.gitignore`, `requirements_streamlit.txt`. **Do not commit** `.streamlit/secrets.toml`; use Streamlit Cloud **Secrets** for production.

---

   - ’
## 2. Supabase (for storing the prompt)

1. Go to [supabase.com](https://supabase.com) and create a free account / project.
2. In the SQL Editor, run:

```sql
create table if not exists app_config (
  key   text primary key,
  value text
);

-- Optional: insert default so first load has a row
-- insert into app_config (key, value) values ('llm_prompt', '')
-- on conflict (key) do nothing;
```

3. In **Project Settings → API**:
   - Copy **Project URL** → you’ll use as `SUPABASE_URL`.
   - Copy **anon public** key → you’ll use as `SUPABASE_KEY`.

---

## 3. Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Choose:
   - **Repository:** your GitHub repo.
   - **Branch:** e.g. `main`.
   - **Main file path:** `streamlit_anomaly_app.py`.
   - **Requirements file:** path to your requirements file (e.g. `requirements_streamlit.txt` or `requirements.txt`).
4. Click **Advanced settings** and add **Secrets** (TOML format).

---

## 4. Secrets (Streamlit Cloud)

In the app’s **Secrets** (e.g. **Settings → Secrets** in the Streamlit Cloud dashboard), add:

```toml
# Required: app login password (team shares this)
APP_PASSWORD = "your_secure_password"

# Required for prompt to persist (Supabase)
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "your_anon_public_key"

# If you use env for Azure OpenAI instead of defaults in code:
# AZURE_OPENAI_API_KEY = "..."
# AZURE_OPENAI_ENDPOINT = "https://..."
# AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
```

- **APP_PASSWORD:** Any user who has this password can open the app (no separate accounts).
- **SUPABASE_URL / SUPABASE_KEY:** So the app can read/write the prompt in Supabase. If these are missing, the app still runs but the prompt is not persisted (only the built-in default is used).

---

## 5. Deploy

1. Click **Deploy**.
2. Wait for the build (install deps, run `streamlit run streamlit_anomaly_app.py`).
3. Open the app URL (e.g. `https://your-app-name.streamlit.app`).
4. Enter **APP_PASSWORD** when prompted; the rest of the app will load.

---

## 6. Improving the prompt (team)

1. Log in with the shared password.
2. In the **sidebar**, open **✏️ Edit prompt**.
3. Change the text (keep placeholders `{TIME_PERIODS_FOR_LLM}` and `{full_row_str}`).
4. Click **Save prompt**. The new prompt is stored in Supabase and used for the next “Analyze” run.

If Supabase is not configured, the sidebar will show a message that the prompt won’t persist; you can still edit and use it for the current session.

---

## 7. Data file (Anomalies_List.csv)

- **Option A:** Commit `Anomalies_List.csv` to the repo so the app finds it at runtime.
- **Option B:** Load from a URL (e.g. Azure Blob, S3, or a shared link) by changing `ANOMALY_CSV` / `load_data()` in the app to fetch from that URL (e.g. `pd.read_csv(url)`).

---

## 8. Local run (optional)

- Create `.streamlit/secrets.toml` (do not commit real secrets):

```toml
APP_PASSWORD = "changeme"
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "your_anon_key"
```

- Run: `streamlit run streamlit_anomaly_app.py`
- Use `requirements_streamlit.txt` (or a `requirements.txt` that includes the same deps).
