"""
erf.i - Empirical Research and Forecasting Institute
Streamlit homepage + predictive-analytics demonstrations.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then create the app at https://share.streamlit.io/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration - edit these values
# --------------------------------------------------------------------------- #
CONFIG = {
    "name": "erf.i",
    "full_name": "Empirical Research and Forecasting Institute",
    "tagline": "Evidence first. Forecasts you can test.",
    "leader": "Dr. Hooshang Lahooti",
    "location": "Sydney, Australia",
    "contact_email": "",   # e.g. "info@yourdomain.org"
    "website": "",         # e.g. "https://www.yourdomain.org"
    "social": "",          # e.g. "https://x.com/yourhandle"
}

CHARCOAL = "#3A3B3F"
BLUE = "#29ABE2"
GREY = "#8A8D93"
LIGHT = "#F4F7FA"

_ASSETS = Path(__file__).parent / "assets"
LOGO_PATH = next((p for p in (_ASSETS / "logo.png", _ASSETS / "logo.jpg") if p.exists()), _ASSETS / "logo.png")

st.set_page_config(
    page_title="erf.i | Empirical Research and Forecasting Institute",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 1150px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .wordmark {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 900;
                 font-size: 4.2rem; letter-spacing: -2px; color: {CHARCOAL}; line-height: 1; }}
    .wordmark .dot {{ color: {BLUE}; }}
    .hero {{ background: linear-gradient(135deg, {CHARCOAL} 0%, #26272a 100%);
             border-radius: 18px; padding: 3rem 2.5rem; color: white; margin: 1rem 0 1.8rem 0; }}
    .hero h1 {{ font-size: 2.5rem; line-height: 1.15; margin: 0 0 .8rem 0; color: white; }}
    .hero h1 span {{ color: {BLUE}; }}
    .hero p {{ font-size: 1.12rem; color: #d7dade; max-width: 760px; margin: 0; }}
    .pill {{ display:inline-block; background:{BLUE}; color:white; border-radius:20px;
             padding:.2rem .85rem; font-size:.78rem; font-weight:600; letter-spacing:.5px;
             margin-bottom:1rem; }}
    .card {{ background:{LIGHT}; border-radius:14px; padding:1.4rem 1.3rem; height:100%;
             border-top:4px solid {BLUE}; }}
    .card h4 {{ margin:0 0 .5rem 0; color:{CHARCOAL}; }}
    .card p {{ margin:0; color:#4a4d54; font-size:.95rem; }}
    .section-title {{ color:{CHARCOAL}; font-weight:800; margin-top:2rem; }}
    .accent {{ color:{BLUE}; }}
    .leader {{ border-left:5px solid {BLUE}; padding:.4rem 1.2rem; background:{LIGHT};
               border-radius:0 12px 12px 0; }}
    .note {{ font-size:.82rem; color:{GREY}; }}
    div[data-testid="stMetric"] {{ background:{LIGHT}; border-radius:12px; padding:.8rem 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_logo(size_rem: float = 4.2) -> None:
    """Show the logo image if present, otherwise a faithful CSS wordmark."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=int(size_rem * 55))
    else:
        st.markdown(
            f'<div class="wordmark" style="font-size:{size_rem}rem">erf<span class="dot">.i</span></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Demo 1: walk-forward forecasting
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def simulate_series(seed: int, noise: float, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")

    # Leading indicator (persistent, mean-reverting)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)

    # Target: trend + seasonality + 2-month-lagged driver + regime shift + AR noise
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, noise)
    regime = np.where(t > n * 0.65, 6.0, 0.0)
    y = 100 + 0.12 * t + 7 * np.sin(2 * np.pi * t / 12) + regime + e
    y[2:] += 1.4 * x[:-2]

    return pd.DataFrame({"date": idx, "y": y, "x": x})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 12):
        f[f"y_lag{k}"] = df["y"].shift(k)
    for k in (1, 2, 3):
        f[f"x_lag{k}"] = df["x"].shift(k)
    m = df["date"].dt.month
    f["m_sin"] = np.sin(2 * np.pi * m / 12)
    f["m_cos"] = np.cos(2 * np.pi * m / 12)
    return f


@st.cache_data(show_spinner=False)
def walk_forward(seed: int, noise: float, start: int = 150, refit_every: int = 6) -> pd.DataFrame:
    df = simulate_series(seed, noise)
    X = make_features(df)
    y = df["y"]
    rows = []
    ridge = hgb = hyb = None
    for t in range(start, len(df)):
        if (t - start) % refit_every == 0:
            tr = slice(13, t)  # only past data; lag-12 needs 12 rows of warm-up
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X.iloc[tr], y.iloc[tr])
            hgb = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.06, max_iter=250, random_state=0
            ).fit(X.iloc[tr], y.iloc[tr])
            resid = y.iloc[tr] - ridge.predict(X.iloc[tr])
            hyb = HistGradientBoostingRegressor(
                max_depth=2, learning_rate=0.05, max_iter=120, random_state=0
            ).fit(X.iloc[tr], resid)
        xt = X.iloc[[t]]
        rows.append(
            {
                "date": df["date"].iloc[t],
                "actual": y.iloc[t],
                "seasonal_naive": y.iloc[t - 12],
                "ridge": float(ridge.predict(xt)[0]),
                "gradient_boosting": float(hgb.predict(xt)[0]),
                "hybrid": float(ridge.predict(xt)[0] + hyb.predict(xt)[0]),
            }
        )
    return pd.DataFrame(rows)


def project_forecasting() -> None:
    st.markdown("### Project 1 · Out-of-sample forecasting with walk-forward validation")
    st.write(
        "A leading-indicator forecasting problem with trend, seasonality, a structural break "
        "and autocorrelated noise. Models are **refit on past data only** and scored on "
        "observations they have never seen, against a seasonal-naive baseline. "
        "Boosting alone cannot extrapolate a trend, which the comparison makes visible; the linear and hybrid "
        "models carry it. A model earns credibility only if it beats the baseline out of sample."
    )
    c1, c2 = st.columns(2)
    noise = c1.slider("Noise level", 0.5, 4.0, 1.5, 0.25, key="fc_noise")
    seed = c2.number_input("Random seed", 0, 9999, 42, key="fc_seed")

    with st.spinner("Running walk-forward validation..."):
        res = walk_forward(int(seed), float(noise))

    base = mean_absolute_error(res["actual"], res["seasonal_naive"])
    scores = {m: mean_absolute_error(res["actual"], res[m]) for m in ("ridge", "hybrid", "gradient_boosting")}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline MAE (seasonal naive)", f"{base:.2f}")
    m2.metric("Ridge MAE", f"{scores['ridge']:.2f}", f"{(1 - scores['ridge'] / base):.0%} skill", delta_color="normal")
    m3.metric("Ridge + boosting MAE", f"{scores['hybrid']:.2f}", f"{(1 - scores['hybrid'] / base):.0%} skill", delta_color="normal")
    m4.metric(
        "Boosting only MAE",
        f"{scores['gradient_boosting']:.2f}",
        f"{(1 - scores['gradient_boosting'] / base):.0%} skill",
        delta_color="normal",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["date"], y=res["actual"], name="Actual", line=dict(color=CHARCOAL, width=2.5)))
    fig.add_trace(go.Scatter(x=res["date"], y=res["ridge"], name="Ridge", line=dict(color=GREY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=res["date"], y=res["hybrid"], name="Ridge + boosting", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(
        go.Scatter(x=res["date"], y=res["gradient_boosting"], name="Boosting only", line=dict(color="#7fd0f0", width=1.3, dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=res["date"], y=res["seasonal_naive"], name="Seasonal naive", line=dict(color="#c9ccd1", width=1.2))
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<p class="note">Skill = 1 − MAE(model) / MAE(baseline). Positive skill means the model beats the baseline. '
        "Data are simulated for demonstration; no real-world series is used.</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Demo 2: early-warning classifier
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def simulate_events(seed: int, n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = -3.4 + 0.6 * X[:, 0] + 0.5 * X[:, 1] + 1.6 * X[:, 0] * X[:, 1] + 1.2 * (X[:, 2] > 1.0) + 0.9 * np.abs(X[:, 3]) - 0.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    df = pd.DataFrame(X, columns=[f"indicator_{i + 1}" for i in range(6)])
    df["event"] = y
    return df


@st.cache_data(show_spinner=False)
def fit_event_models(seed: int) -> dict:
    df = simulate_events(seed)
    cut = int(len(df) * 0.7)  # chronological split, no shuffling
    feats = [c for c in df.columns if c != "event"]
    tr, te = df.iloc[:cut], df.iloc[cut:]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[feats], tr["event"])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250, random_state=0).fit(
        tr[feats], tr["event"]
    )
    return {
        "y_test": te["event"].to_numpy(),
        "Logistic regression": lr.predict_proba(te[feats])[:, 1],
        "Gradient boosting": gb.predict_proba(te[feats])[:, 1],
        "base_rate": float(te["event"].mean()),
    }


def project_early_warning() -> None:
    st.markdown("### Project 2 · Early-warning model for rare events")
    st.write(
        "A rare-event detection problem with non-linear and interaction effects. "
        "Models are trained on the first 70% of the timeline and evaluated on the last 30%. "
        "Beyond ranking quality (ROC-AUC, average precision) we check **calibration**, "
        "because a probability is only useful if it means what it says."
    )
    seed = st.number_input("Random seed", 0, 9999, 7, key="ew_seed")
    out = fit_event_models(int(seed))
    y = out["y_test"]

    rows = []
    for m in ("Logistic regression", "Gradient boosting"):
        rows.append(
            {
                "Model": m,
                "ROC-AUC": roc_auc_score(y, out[m]),
                "Average precision": average_precision_score(y, out[m]),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format({"ROC-AUC": "{:.3f}", "Average precision": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Out-of-sample event base rate: {out['base_rate']:.1%} (average precision of a random model ≈ base rate).")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for m, col in (("Logistic regression", GREY), ("Gradient boosting", BLUE)):
            fpr, tpr, _ = roc_curve(y, out[m])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=m, line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="#c9ccd1", dash="dash")))
        fig.update_layout(
            template="plotly_white", height=360, title="ROC curve",
            xaxis_title="False positive rate", yaxis_title="True positive rate",
            margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        for m, col in (("Logistic regression", GREY), ("Gradient boosting", BLUE)):
            frac, mean_pred = calibration_curve(y, out[m], n_bins=8, strategy="quantile")
            fig.add_trace(go.Scatter(x=mean_pred, y=frac, name=m, mode="lines+markers", line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], name="Perfect", line=dict(color="#c9ccd1", dash="dash")))
        fig.update_layout(
            template="plotly_white", height=360, title="Calibration",
            xaxis_title="Predicted probability", yaxis_title="Observed frequency",
            margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Alert threshold trade-off** (gradient boosting)")
    thr = st.slider("Alert when predicted probability exceeds", 0.02, 0.80, 0.20, 0.01)
    pred = out["Gradient boosting"] >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    a, b, c = st.columns(3)
    a.metric("Alerts issued", f"{int(pred.sum())}")
    b.metric("Precision", f"{precision:.0%}")
    c.metric("Recall (events caught)", f"{recall:.0%}")
    st.markdown(
        '<p class="note">Data are simulated for demonstration. The point is the method: time-ordered validation, '
        "baselines, calibration and explicit decision thresholds.</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">DATA SCIENCE · FORECASTING · EVIDENCE</span>
            <h1>Turning data into <span>testable forecasts</span>.</h1>
            <p>{CONFIG['full_name']} (erf.i) applies rigorous data science to build predictive models
            whose accuracy is measured out of sample, reported openly and compared with honest baselines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h3 class="section-title">What we <span class="accent">do</span></h3>', unsafe_allow_html=True)
    cols = st.columns(4)
    pillars = [
        ("Empirical research", "Transparent, reproducible analysis built on data rather than narrative."),
        ("Predictive modelling", "Statistical and machine-learning models from regularised regression to gradient boosting."),
        ("Forecast evaluation", "Walk-forward validation, baselines and calibration. Every claim is scored."),
        ("Decision support", "Probabilities and thresholds translated into clear, actionable signals."),
    ]
    for col, (h, p) in zip(cols, pillars):
        col.markdown(f'<div class="card"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

    st.markdown('<h3 class="section-title">Our <span class="accent">standard</span></h3>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Validation", "Out-of-sample")
    s2.metric("Benchmark", "Always a baseline")
    s3.metric("Reporting", "Uncertainty included")

    st.markdown('<h3 class="section-title">Featured <span class="accent">demonstrations</span></h3>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.markdown(
        '<div class="card"><h4>Walk-forward forecasting</h4>'
        "<p>Leading-indicator model that is refit through time and scored against a seasonal baseline. "
        "Open the <b>Projects</b> tab to run it live.</p></div>",
        unsafe_allow_html=True,
    )
    d2.markdown(
        '<div class="card"><h4>Rare-event early warning</h4>'
        "<p>Calibrated probability model with an adjustable alert threshold and a precision/recall trade-off. "
        "Open the <b>Projects</b> tab to run it live.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<h3 class="section-title">Leadership</h3>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="leader">
            <b style="font-size:1.1rem">{CONFIG['leader']}</b><br>
            Leads {CONFIG['name']} · {CONFIG['location']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_projects() -> None:
    st.markdown("## Projects")
    st.write("Interactive demonstrations of the institute's approach. Adjust the controls and the models re-run live.")
    tab1, tab2 = st.tabs(["Forecasting", "Early warning"])
    with tab1:
        project_forecasting()
    with tab2:
        project_early_warning()


def page_about() -> None:
    st.markdown("## About")
    st.write(
        f"**{CONFIG['full_name']} (erf.i)** is a data-science institute focused on empirical research and "
        "forecasting. Its work rests on three commitments:"
    )
    st.markdown(
        "- **Evidence over assertion**: conclusions follow from data and are reproducible.\n"
        "- **Honest evaluation**: forecasts are tested out of sample against simple baselines.\n"
        "- **Clarity**: results are communicated with their uncertainty, in plain language."
    )
    st.markdown(f"**Leadership:** {CONFIG['leader']}  \n**Base:** {CONFIG['location']}")
    links = []
    if CONFIG["contact_email"]:
        links.append(f"[{CONFIG['contact_email']}](mailto:{CONFIG['contact_email']})")
    if CONFIG["website"]:
        links.append(f"[Website]({CONFIG['website']})")
    if CONFIG["social"]:
        links.append(f"[Social]({CONFIG['social']})")
    if links:
        st.markdown("**Contact:** " + " · ".join(links))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
head_l, head_r = st.columns([1, 3])
with head_l:
    render_logo()
with head_r:
    st.markdown(
        f'<div style="padding-top:1.1rem;color:{GREY};font-size:.95rem;text-align:right">'
        f"{CONFIG['full_name']}<br>{CONFIG['location']}</div>",
        unsafe_allow_html=True,
    )

nav_home, nav_projects, nav_about = st.tabs(["Home", "Projects", "About"])
with nav_home:
    page_home()
with nav_projects:
    page_projects()
with nav_about:
    page_about()

st.markdown("---")
st.markdown(
    f'<p class="note">© {pd.Timestamp.today().year} {CONFIG["name"]} · {CONFIG["full_name"]}. '
    "Demonstrations use simulated data and are for illustration only; they are not forecasts or advice.</p>",
    unsafe_allow_html=True,
)
