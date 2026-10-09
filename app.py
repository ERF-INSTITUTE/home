"""
erf.i - Empirical Research and Forecasting Institute
Streamlit homepage + predictive-analytics demonstrations.

Run locally:   streamlit run app.py
Deploy:        push to GitHub (include the hidden .streamlit/ folder), then create
               the app at https://share.streamlit.io/
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
    "leader": "Dr. Hooshang Lahooti",
    "location": "Sydney, Australia",
    "contact_email": "",   # e.g. "info@yourdomain.org"
    "website": "",         # e.g. "https://www.yourdomain.org"
    "social": "",          # e.g. "https://x.com/yourhandle"
}

# Palette: white, gray, logo blue, one small orange accent
INK = "#666666"        # body text (gray)
INK_DARK = "#4D4D4D"   # headings, slightly stronger for hierarchy
MUTED = "#9A9DA2"      # captions
LINE = "#E6E8EB"       # hairlines
BLUE = "#29ABE2"       # logo blue
ORANGE = "#F28C28"     # accent, used sparingly
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

_ASSETS = Path(__file__).parent / "assets"
LOGO_PATH = next((p for p in (_ASSETS / "logo.png", _ASSETS / "logo.jpg") if p.exists()), _ASSETS / "logo.png")

st.set_page_config(
    page_title="erf.i | Empirical Research and Forecasting Institute",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Styling (forces a clean white page whatever the viewer's system theme is)
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stMain"], .main {{ background:#FFFFFF !important; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stMarkdown {{
    font-family:{FONT}; color:{INK};
}}
.block-container {{ max-width:1100px; padding-top:2.2rem; padding-bottom:3rem; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility:hidden; }}

h1, h2, h3, h4 {{ font-family:{FONT} !important; color:{INK_DARK} !important; letter-spacing:-0.02em; }}
.stApp a {{ color:{BLUE}; }}

/* tabs */
button[data-baseweb="tab"] {{ padding:.6rem 1.1rem .6rem 0; margin-right:1.2rem; background:transparent !important; }}
button[data-baseweb="tab"] p {{ font-size:1rem; font-weight:600; color:{MUTED} !important; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color:{INK_DARK} !important; }}
[data-baseweb="tab-highlight"] {{ background:{BLUE} !important; height:3px; }}
[data-baseweb="tab-border"] {{ background:{LINE} !important; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="base-input"] {{
    background:#FFFFFF !important; color:{INK} !important; }}
[data-baseweb="input"] {{ border:1px solid {LINE} !important; border-radius:8px; }}
[data-testid="stNumberInput"] button {{ background:#FFFFFF !important; color:{INK} !important; border-color:{LINE} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background:{BLUE} !important; box-shadow:none !important; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{ color:{INK} !important; }}

/* identity */
.eyebrow {{ font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:{BLUE}; }}
.hero-title {{ font-size:3.1rem; font-weight:800; line-height:1.08; letter-spacing:-0.03em;
               color:{INK_DARK}; margin:.6rem 0 1.1rem 0; }}
.hero-title em {{ font-style:normal; color:{BLUE}; }}
.lede {{ font-size:1.15rem; line-height:1.65; color:{INK}; max-width:720px; }}
.rule {{ border:0; border-top:1px solid {LINE}; margin:2.4rem 0 1.6rem 0; }}
.sec {{ font-size:1.5rem; font-weight:700; color:{INK_DARK}; margin:0 0 1.2rem 0; letter-spacing:-0.02em; }}
.sec::after {{ content:""; display:block; width:34px; height:3px; background:{ORANGE}; margin-top:.55rem; border-radius:2px; }}

.pillar h4 {{ font-size:1.05rem; font-weight:700; margin:0 0 .4rem 0; color:{INK_DARK} !important; }}
.pillar p {{ font-size:.95rem; line-height:1.6; margin:0; }}
.pillar .n {{ font-size:.8rem; font-weight:700; color:{BLUE}; letter-spacing:.1em; margin-bottom:.5rem; }}

.panel {{ border:1px solid {LINE}; border-radius:10px; padding:1.5rem 1.5rem 1.3rem 1.5rem; background:#FFFFFF; }}
.panel h4 {{ font-size:1.1rem; font-weight:700; margin:0 0 .5rem 0; color:{INK_DARK} !important; }}
.panel p {{ font-size:.95rem; line-height:1.6; margin:0 0 1rem 0; }}
.panel .big {{ font-size:2.1rem; font-weight:800; color:{BLUE}; letter-spacing:-0.02em; line-height:1; }}
.panel .cap {{ font-size:.82rem; color:{MUTED}; margin-top:.3rem; }}

.kpi {{ border-top:2px solid {LINE}; padding-top:.7rem; }}
.kpi.hi {{ border-top-color:{BLUE}; }}
.kpi.warn {{ border-top-color:{ORANGE}; }}
.kpi .l {{ font-size:.74rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; }}
.kpi .v {{ font-size:1.9rem; font-weight:800; color:{INK_DARK}; letter-spacing:-0.02em; line-height:1.15; }}
.kpi .d {{ font-size:.85rem; font-weight:600; color:{INK}; }}
.kpi.hi .d {{ color:{BLUE}; }}
.kpi.warn .d {{ color:{ORANGE}; }}

.leader {{ border-left:3px solid {BLUE}; padding:.2rem 0 .2rem 1.1rem; }}
.leader b {{ font-size:1.1rem; color:{INK_DARK}; }}
.note {{ font-size:.82rem; color:{MUTED}; line-height:1.5; }}

table.t {{ border-collapse:collapse; width:100%; font-size:.95rem; }}
table.t th {{ text-align:left; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase;
              color:{MUTED}; font-weight:600; padding:.55rem .5rem; border-bottom:2px solid {LINE}; }}
table.t td {{ padding:.65rem .5rem; border-bottom:1px solid {LINE}; color:{INK}; }}
table.t td.best {{ color:{BLUE}; font-weight:700; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def kpi(label: str, value: str, delta: str = "", kind: str = "") -> str:
    d = f'<div class="d">{delta}</div>' if delta else '<div class="d">&nbsp;</div>'
    return f'<div class="kpi {kind}"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="rule">', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=title, font=dict(size=15, color=INK_DARK), x=0),
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        legend=dict(orientation="h", y=1.12 if not title else -0.25, x=0),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor="#F0F1F3", linecolor=LINE, zeroline=False)
    return fig


def html_table(df: pd.DataFrame, best_cols: list[str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                cls = ' class="best"' if c in best_cols and v == df[c].max() else ""
                cells += f"<td{cls}>{v:.3f}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    return f'<table class="t"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def header() -> None:
    left, right = st.columns([1, 3], vertical_alignment="center")
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=96)
        else:
            st.markdown(
                f'<div style="font-size:3rem;font-weight:900;letter-spacing:-2px;color:{INK_DARK}">'
                f'erf<span style="color:{BLUE}">.i</span></div>',
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f'<div style="text-align:right;font-size:.9rem;color:{MUTED};line-height:1.5">'
            f"{CONFIG['full_name']}<br>{CONFIG['location']}</div>",
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

    x = np.zeros(n)  # leading indicator (persistent, mean-reverting)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)

    e = np.zeros(n)  # autocorrelated noise
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, noise)
    regime = np.where(t > n * 0.65, 6.0, 0.0)  # structural break
    y = 100 + 0.12 * t + 7 * np.sin(2 * np.pi * t / 12) + regime + e
    y[2:] += 1.4 * x[:-2]  # indicator leads the target by two periods

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
            tr = slice(13, t)  # past data only
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X.iloc[tr], y.iloc[tr])
            hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=250, random_state=0).fit(
                X.iloc[tr], y.iloc[tr]
            )
            resid = y.iloc[tr] - ridge.predict(X.iloc[tr])
            hyb = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=120, random_state=0).fit(
                X.iloc[tr], resid
            )
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
    st.markdown("### Out-of-sample forecasting with walk-forward validation")
    st.write(
        "A leading-indicator problem with trend, seasonality, a structural break and autocorrelated noise. "
        "Models are refit on past data only and scored on observations they have never seen, against a "
        "seasonal-naive baseline. Boosting alone cannot extrapolate a trend, which the comparison makes visible; "
        "the linear and hybrid models carry it. A model earns credibility only if it beats the baseline out of sample."
    )
    c1, c2, _ = st.columns([2, 1, 1])
    noise = c1.slider("Noise level", 0.5, 4.0, 1.5, 0.25, key="fc_noise")
    seed = c2.number_input("Random seed", 0, 9999, 42, key="fc_seed")

    with st.spinner("Running walk-forward validation..."):
        res = walk_forward(int(seed), float(noise))

    base = mean_absolute_error(res["actual"], res["seasonal_naive"])
    sc = {m: mean_absolute_error(res["actual"], res[m]) for m in ("ridge", "hybrid", "gradient_boosting")}
    skill = {m: 1 - v / base for m, v in sc.items()}

    k = st.columns(4)
    k[0].markdown(kpi("Baseline MAE", f"{base:.2f}", "seasonal naive"), unsafe_allow_html=True)
    k[1].markdown(kpi("Ridge MAE", f"{sc['ridge']:.2f}", f"{skill['ridge']:.0%} better than baseline", "hi"), unsafe_allow_html=True)
    k[2].markdown(kpi("Ridge + boosting MAE", f"{sc['hybrid']:.2f}", f"{skill['hybrid']:.0%} better than baseline", "hi"), unsafe_allow_html=True)
    k[3].markdown(
        kpi("Boosting only MAE", f"{sc['gradient_boosting']:.2f}", f"{skill['gradient_boosting']:.0%} better than baseline", "warn"),
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["date"], y=res["seasonal_naive"], name="Seasonal naive", line=dict(color="#C9CDD2", width=1.2)))
    fig.add_trace(go.Scatter(x=res["date"], y=res["gradient_boosting"], name="Boosting only", line=dict(color=ORANGE, width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=res["date"], y=res["hybrid"], name="Ridge + boosting", line=dict(color=BLUE, width=2.6)))
    fig.add_trace(go.Scatter(x=res["date"], y=res["actual"], name="Actual", line=dict(color=INK, width=2)))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    st.markdown(
        '<p class="note">Improvement = 1 − MAE(model) ÷ MAE(baseline). Data are simulated for demonstration; '
        "no real-world series is used.</p>",
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
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame(X, columns=[f"indicator_{i + 1}" for i in range(6)])
    df["event"] = y
    return df


@st.cache_data(show_spinner=False)
def fit_event_models(seed: int) -> dict:
    df = simulate_events(seed)
    cut = int(len(df) * 0.7)  # time-ordered split, no shuffling
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
    st.markdown("### Early-warning model for rare events")
    st.write(
        "A rare-event problem with non-linear and interaction effects. Models are trained on the first 70% of the "
        "timeline and evaluated on the last 30%. Beyond ranking quality (ROC-AUC, average precision) we check "
        "calibration, because a probability is only useful if it means what it says."
    )
    c1, _ = st.columns([1, 3])
    seed = c1.number_input("Random seed", 0, 9999, 7, key="ew_seed")
    out = fit_event_models(int(seed))
    y = out["y_test"]

    tbl = pd.DataFrame(
        [
            {
                "Model": m,
                "ROC-AUC": roc_auc_score(y, out[m]),
                "Average precision": average_precision_score(y, out[m]),
            }
            for m in ("Logistic regression", "Gradient boosting")
        ]
    )
    st.markdown(html_table(tbl, ["ROC-AUC", "Average precision"]), unsafe_allow_html=True)
    st.markdown(
        f'<p class="note" style="margin-top:.6rem">Out-of-sample event base rate: {out["base_rate"]:.1%}. '
        "A random model scores an average precision close to this base rate.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for m, col in (("Logistic regression", "#B5B9BF"), ("Gradient boosting", BLUE)):
            fpr, tpr, _ = roc_curve(y, out[m])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=m, line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="#D5D8DC", dash="dash")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig, 360, "ROC curve"), use_container_width=True)
    with right:
        fig = go.Figure()
        for m, col in (("Logistic regression", "#B5B9BF"), ("Gradient boosting", BLUE)):
            frac, mean_pred = calibration_curve(y, out[m], n_bins=8, strategy="quantile")
            fig.add_trace(go.Scatter(x=mean_pred, y=frac, name=m, mode="lines+markers", line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], name="Perfect", line=dict(color="#D5D8DC", dash="dash")))
        fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        st.plotly_chart(style_fig(fig, 360, "Calibration"), use_container_width=True)

    st.markdown("**Alert threshold trade-off** (gradient boosting)")
    thr = st.slider("Alert when predicted probability exceeds", 0.02, 0.80, 0.20, 0.01)
    pred = out["Gradient boosting"] >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    k = st.columns(3)
    k[0].markdown(kpi("Alerts issued", f"{int(pred.sum())}", f"of {len(y)} periods"), unsafe_allow_html=True)
    k[1].markdown(kpi("Precision", f"{precision:.0%}", "alerts that were real events", "hi"), unsafe_allow_html=True)
    k[2].markdown(kpi("Recall", f"{recall:.0%}", "events caught", "warn"), unsafe_allow_html=True)
    st.markdown(
        '<p class="note" style="margin-top:1rem">Data are simulated for demonstration. The point is the method: '
        "time-ordered validation, baselines, calibration and explicit decision thresholds.</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.markdown(
        f"""<div style="padding:2.2rem 0 .4rem 0">
<div class="eyebrow">Data science · Forecasting · Evidence</div>
<div class="hero-title">Turning data into <em>testable forecasts</em>.</div>
<div class="lede">{CONFIG['full_name']} applies rigorous data science to build predictive models whose accuracy
is measured out of sample, reported openly and compared with honest baselines.</div>
</div>""",
        unsafe_allow_html=True,
    )
    rule()

    section("What we do")
    pillars = [
        ("Empirical research", "Transparent, reproducible analysis built on data rather than narrative."),
        ("Predictive modelling", "Statistical and machine-learning models, from regularised regression to gradient boosting."),
        ("Forecast evaluation", "Walk-forward validation, baselines and calibration. Every claim is scored."),
        ("Decision support", "Probabilities and thresholds translated into clear, actionable signals."),
    ]
    for col, (i, (h, p)) in zip(st.columns(4, gap="large"), enumerate(pillars, 1)):
        col.markdown(f'<div class="pillar"><div class="n">0{i}</div><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)
    rule()

    section("Demonstrated predictive power")
    fc = walk_forward(42, 1.5)
    base = mean_absolute_error(fc["actual"], fc["seasonal_naive"])
    best = mean_absolute_error(fc["actual"], fc["hybrid"])
    ev = fit_event_models(7)
    auc_gb = roc_auc_score(ev["y_test"], ev["Gradient boosting"])
    auc_lr = roc_auc_score(ev["y_test"], ev["Logistic regression"])

    a, b = st.columns(2, gap="large")
    a.markdown(
        f'<div class="panel"><h4>Walk-forward forecasting</h4>'
        "<p>A leading-indicator model, refit through time and scored only on data it has never seen, against a seasonal baseline.</p>"
        f'<div class="big">{1 - best / base:.0%}</div><div class="cap">lower forecast error than the baseline, out of sample</div></div>',
        unsafe_allow_html=True,
    )
    b.markdown(
        f'<div class="panel"><h4>Rare-event early warning</h4>'
        "<p>A calibrated probability model with an adjustable alert threshold and an explicit precision/recall trade-off.</p>"
        f'<div class="big">{auc_gb:.2f}</div><div class="cap">out-of-sample ROC-AUC, against {auc_lr:.2f} for a linear baseline</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="note" style="margin-top:.8rem">Simulated data. Open the Projects tab to run both demonstrations live.</p>',
        unsafe_allow_html=True,
    )
    rule()

    section("Our standard")
    std = [
        ("Out-of-sample validation", "Models are judged only on data they have not seen."),
        ("Always a baseline", "A forecast has value only relative to a simple alternative."),
        ("Uncertainty reported", "Calibration and error are shown next to every result."),
    ]
    for col, (h, p) in zip(st.columns(3, gap="large"), std):
        col.markdown(f'<div class="pillar"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)
    rule()

    section("Leadership")
    st.markdown(
        f'<div class="leader"><b>{CONFIG["leader"]}</b><br>Leads {CONFIG["name"]} · {CONFIG["location"]}</div>',
        unsafe_allow_html=True,
    )


def page_projects() -> None:
    st.markdown('<div class="sec">Projects</div>', unsafe_allow_html=True)
    st.write("Interactive demonstrations of the institute's approach. Adjust the controls and the models re-run live.")
    t1, t2 = st.tabs(["Forecasting", "Early warning"])
    with t1:
        project_forecasting()
    with t2:
        project_early_warning()


def page_about() -> None:
    st.markdown('<div class="sec">About</div>', unsafe_allow_html=True)
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
header()
tab_home, tab_projects, tab_about = st.tabs(["Home", "Projects", "About"])
with tab_home:
    page_home()
with tab_projects:
    page_projects()
with tab_about:
    page_about()

rule()
st.markdown(
    f'<p class="note">© {pd.Timestamp.today().year} {CONFIG["name"]} · {CONFIG["full_name"]}. '
    "Demonstrations use simulated data and are for illustration only; they are not forecasts or advice.</p>",
    unsafe_allow_html=True,
)
