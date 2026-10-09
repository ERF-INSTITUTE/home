# erf.i · Empirical Research and Forecasting Institute

Streamlit homepage and predictive-analytics demonstrations for erf.i.

## Structure

```
app.py                  # the app
requirements.txt
.streamlit/config.toml  # brand theme
assets/logo.png (or logo.jpg)  # add your logo here (optional, CSS wordmark is the fallback)
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository under the official erf.i account and push these files.
2. Save the logo as `assets/logo.png` (square PNG, transparent or white background).
3. Go to https://share.streamlit.io/, click **Create app**, pick the repository and branch, set the main file to `app.py`, and deploy.
4. Edit the `CONFIG` block at the top of `app.py` to add the contact email, website and social link.

## Notes

Both demonstrations use simulated data so results are reproducible and no third-party data is involved.
