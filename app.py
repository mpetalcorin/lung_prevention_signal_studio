import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from scipy.special import expit

st.set_page_config(page_title="Lung Prevention Signal Studio", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

PROTEINS = [
    "CEACAM5", "WFDC2", "LAMP3", "GDF15", "CXCL17", "ALPP", "MMP12",
    "PIGR", "PLAUR", "PRSS8", "SFTPD", "CDCP1", "SFTPA1", "TNFSF13B"
]

PROTEIN_GROUPS = {
    "Inflammation / immune signalling": ["CXCL17", "CDCP1", "GDF15", "PIGR", "TNFSF13B", "PLAUR"],
    "Matrix remodelling": ["MMP12"],
    "Epithelial secretion / shedding": ["CEACAM5", "WFDC2", "ALPP", "PRSS8"],
    "Surfactant / AT2 programme": ["LAMP3", "SFTPD", "SFTPA1"],
}

RR_ANCHOR = {
    "CEACAM5": 2.35, "WFDC2": 2.20, "LAMP3": 1.95, "GDF15": 1.80, "CXCL17": 1.65,
    "ALPP": 1.55, "MMP12": 1.48, "PIGR": 1.42, "PLAUR": 1.38, "PRSS8": 1.30,
    "SFTPD": 1.24, "CDCP1": 1.20, "SFTPA1": 1.16, "TNFSF13B": 1.12,
}

CELL_PROGRAMS = {
    "AT2 / surfactant": ["SFTPA1", "SFTPD", "LAMP3", "CXCL17"],
    "Secretory airway": ["WFDC2", "PIGR", "PRSS8", "CEACAM5"],
    "Myeloid inflammatory": ["PLAUR", "MMP12", "GDF15", "TNFSF13B", "CDCP1"],
    "KAC transitional": ["KRT8", "CLDN4", "KRT18", "LGALS3", "CEACAM5", "WFDC2"],
}

BASE_COLORWAY = ["#00D4FF", "#FF4ECD", "#FFB000", "#33F2A0", "#9D7BFF", "#FF4B4B", "#64FFDA", "#FF7A00"]
PALETTES = {
    "Neon translational": ["#00D4FF", "#FF4ECD", "#FFB000", "#33F2A0", "#9D7BFF", "#FF4B4B", "#64FFDA", "#FF7A00"],
    "Clinical light": ["#1663B7", "#00A6A6", "#F59E0B", "#7C3AED", "#EF4444", "#0F766E", "#6B7280", "#D946EF"],
    "Nature journal": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#999999"],
    "High contrast": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
}
CONTINUOUS_SCALES = ["Turbo", "Viridis", "Plasma", "Cividis", "Bluered", "RdBu"]

with st.sidebar:
    st.markdown("## 🎨 Appearance")
    appearance_mode = st.radio("Theme mode", ["Dark mode", "Light mode"], horizontal=True)
    palette_name = st.selectbox("Categorical palette", list(PALETTES), index=0)
    continuous_scale = st.selectbox("Continuous colour scale", CONTINUOUS_SCALES, index=0)
    transparent_plots = st.toggle("Transparent plot panels", value=True)
    compact_cards = st.toggle("Compact metric cards", value=False)

COLORWAY = PALETTES[palette_name]
CONTINUOUS_SCALE = continuous_scale
PLOTLY_TEMPLATE = "plotly_white" if appearance_mode == "Light mode" else "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)" if transparent_plots else ("#FFFFFF" if appearance_mode == "Light mode" else "#0B1026")
PAPER_BG = "rgba(0,0,0,0)" if transparent_plots else ("#FFFFFF" if appearance_mode == "Light mode" else "#0B1026")
CARD_PAD = "12px" if compact_cards else "16px"

if appearance_mode == "Light mode":
    app_bg = "linear-gradient(135deg, #F8FBFF 0%, #EEF7FF 42%, #FFF8F2 100%)"
    sidebar_bg = "linear-gradient(180deg, #FFFFFF 0%, #EAF4FF 100%)"
    text_color = "#102033"
    heading_color = "#071A2D"
    note_color = "#43566B"
    card_bg = "linear-gradient(135deg, rgba(22,99,183,0.10), rgba(0,166,166,0.08), rgba(245,158,11,0.09))"
    metric_bg = "linear-gradient(135deg, rgba(22,99,183,0.13), rgba(0,166,166,0.10))"
    border = "rgba(16,32,51,0.16)"
    shadow = "0 10px 28px rgba(16,32,51,0.10)"
else:
    app_bg = "radial-gradient(circle at top left, #1A2B68 0, #070B2A 34%, #050516 100%)"
    sidebar_bg = "linear-gradient(180deg, #071039 0%, #070B2A 100%)"
    text_color = "#F7FBFF"
    heading_color = "#F7FBFF"
    note_color = "#CFE6FF"
    card_bg = "linear-gradient(135deg, rgba(0,212,255,0.11), rgba(157,123,255,0.12), rgba(255,78,205,0.10))"
    metric_bg = "linear-gradient(135deg, rgba(0,212,255,0.18), rgba(255,78,205,0.13))"
    border = "rgba(255,255,255,0.15)"
    shadow = "0 12px 35px rgba(0,0,0,0.25)"

st.markdown(f"""
<style>
.stApp {{background: {app_bg}; color: {text_color};}}
h1, h2, h3, h4 {{color: {heading_color};}}
p, li, label, span, div {{color: inherit;}}
section[data-testid="stSidebar"] {{background: {sidebar_bg};}}
section[data-testid="stSidebar"] * {{color: {text_color};}}
div[data-testid="stMetric"] {{background: {metric_bg}; border: 1px solid {border}; padding: {CARD_PAD}; border-radius: 18px; box-shadow: {shadow};}}
.glass {{background: {card_bg}; border: 1px solid {border}; border-radius: 22px; padding: 18px 20px; box-shadow: {shadow};}}
.pill {{display: inline-block; padding: 0.35rem 0.7rem; margin: 0.15rem; border-radius: 999px; background: {metric_bg}; border: 1px solid {border}; font-size: 0.85rem;}}
.small-note {{color: {note_color}; font-size: 0.92rem;}}
.big-card {{background: {card_bg}; border: 1px solid {border}; border-radius: 26px; padding: 22px; box-shadow: {shadow};}}
.stTabs [data-baseweb="tab-list"] {{gap: 0.35rem;}}
.stTabs [data-baseweb="tab"] {{border-radius: 999px; padding: 0.45rem 0.8rem; background: {metric_bg}; border: 1px solid {border};}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def simulate_cohort(n=5000, seed=42, pm_scale=1.0, case_multiplier=1.0):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(61, 8.5, n), 40, 85)
    smoking_status = rng.choice(["Never", "Former", "Current"], size=n, p=[0.35, 0.43, 0.22])
    smoke_code = pd.Series(smoking_status).map({"Never": 0, "Former": 1, "Current": 2}).values
    pack_years = np.where(smoke_code == 0, rng.gamma(0.6, 1.2, n), rng.gamma(2.2 + smoke_code, 7.0, n))
    pack_years = np.clip(pack_years, 0, 90)
    copd = rng.binomial(1, expit(-4.0 + 0.032 * (age - 50) + 0.026 * pack_years + 0.65 * (smoke_code == 2)))
    pm25 = np.clip(rng.lognormal(mean=np.log(10 * pm_scale), sigma=0.35, size=n), 3, 38)
    il1b_axis = np.clip(0.30 * copd + 0.16 * smoke_code + 0.045 * (pm25 - 10) + rng.normal(0, 0.45, n), -1.5, 3.5)
    egfr_clone_proxy = rng.binomial(1, expit(-3.6 + 0.020 * (age - 50) + 0.025 * np.maximum(pm25 - 8, 0)))
    sex = rng.choice(["Female", "Male"], n, p=[0.53, 0.47])

    signature_latent = 0.55 * il1b_axis + 0.22 * smoke_code + 0.035 * (pm25 - 10) + 0.30 * copd + 0.35 * egfr_clone_proxy
    data = {"age": age, "sex": sex, "smoking_status": smoking_status, "smoke_code": smoke_code, "pack_years": pack_years, "COPD": copd, "PM2_5": pm25, "IL1B_axis": il1b_axis, "EGFR_clone_proxy": egfr_clone_proxy}
    for p in PROTEINS:
        weight = np.log(RR_ANCHOR[p])
        data[p] = rng.normal(0, 0.65, n) + weight * signature_latent + rng.normal(0, 0.15, n)
    df = pd.DataFrame(data)
    df["signature_score"] = df[PROTEINS].mean(axis=1)
    df["KAC_index"] = expit(-0.65 + 0.80 * df["EGFR_clone_proxy"] + 0.72 * df["IL1B_axis"] + 0.045 * (df["PM2_5"] - 10) + 0.75 * df["signature_score"])
    base_risk = -6.6 + np.log(case_multiplier) + 0.036 * (age - 55) + 0.020 * pack_years + 0.55 * (smoke_code == 2) + 0.42 * copd + 0.055 * (pm25 - 10) + 0.42 * egfr_clone_proxy
    logit = base_risk + 0.72 * df["signature_score"].values + 0.45 * df["KAC_index"].values + rng.normal(0, 0.25, n)
    df["risk_true"] = expit(logit)
    df["future_lung_cancer"] = rng.binomial(1, df["risk_true"])
    df["risk_tier"] = pd.qcut(df["signature_score"], q=[0, .5, .8, .95, 1.0], labels=["Low", "Intermediate", "High", "Very high"])
    df["anti_IL1B_response_index"] = expit(-0.4 + 1.35 * df["signature_score"] + 0.55 * df["IL1B_axis"])
    return df

@st.cache_resource(show_spinner=False)
def train_model(_df, n, seed, pm_scale, case_multiplier):
    features = ["age", "smoke_code", "pack_years", "COPD", "PM2_5", "IL1B_axis", "EGFR_clone_proxy", "KAC_index"] + PROTEINS
    X = _df[features]
    y = _df["future_lung_cancer"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=7, stratify=y)
    model = GradientBoostingClassifier(random_state=7, n_estimators=220, learning_rate=0.032, max_depth=3)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred) if len(np.unique(y_test)) > 1 else np.nan
    ap = average_precision_score(y_test, pred)
    brier = brier_score_loss(y_test, pred)
    return model, features, X_train, X_test, y_train, y_test, pred, auc, ap, brier

@st.cache_data(show_spinner=False)
def simulate_timecourses(n_people=240, seed=2026):
    rng = np.random.default_rng(seed + 19)
    rows = []
    times = np.array([-60, -48, -36, -24, -12, 0])
    for i in range(n_people):
        is_case = i < n_people // 2
        base = rng.normal(0, 0.35)
        slope = rng.normal(0.003, 0.002) + (0.018 if is_case else 0.002)
        accel = 0.00022 if is_case else 0.00004
        for t in times:
            months_to_dx = -t
            signal = base + slope * (60 - months_to_dx) + accel * (60 - months_to_dx)**2 + rng.normal(0, 0.18)
            rows.append({"person": i, "case_status": "Future lung cancer" if is_case else "Control", "months_to_diagnosis": months_to_dx, "signature_score": signal})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def simulate_single_cell(seed=42, n=5200):
    rng = np.random.default_rng(seed + 101)
    cell_types = ["AT2", "Secretory", "Club", "Basal", "KAC", "Macrophage", "Monocyte", "Fibroblast", "Endothelium", "T cell"]
    probs = np.array([0.18, 0.12, 0.09, 0.07, 0.06, 0.16, 0.08, 0.10, 0.08, 0.06])
    ct = rng.choice(cell_types, size=n, p=probs / probs.sum())
    centers = {"AT2":(-2.4, 1.8), "Secretory":(0.0, 2.0), "Club":(1.8, 1.4), "Basal":(2.4, -0.6), "KAC":(-0.5, 0.1), "Macrophage":(-2.1, -1.4), "Monocyte":(-1.0, -2.2), "Fibroblast":(1.0, -2.0), "Endothelium":(2.8, 1.0), "T cell":(-3.0, 0.0)}
    rows = []
    for c in ct:
        x0, y0 = centers[c]
        x = rng.normal(x0, 0.38)
        y = rng.normal(y0, 0.38)
        sig = rng.normal(0.2, 0.17)
        if c in ["AT2", "Secretory", "KAC"]:
            sig += rng.normal(1.0, 0.24)
        if c in ["Macrophage", "Monocyte", "Fibroblast"]:
            sig += rng.normal(0.62, 0.24)
        kac = expit(-1.4 + (1.9 if c == "KAC" else 0) + (0.5 if c == "AT2" else 0) + rng.normal(0, .5))
        rows.append({"UMAP1": x, "UMAP2": y, "cell_type": c, "signature_score": sig, "KAC_score": kac})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def external_validation(seed=2026):
    rng = np.random.default_rng(seed + 303)
    cohorts = ["UKBB hold-out", "UKCTOCS", "EPIC", "TALENT", "CANTOS", "China Kadoorie", "Iceland", "US prospective"]
    rows = []
    for p in PROTEINS:
        anchor = RR_ANCHOR[p]
        for c in cohorts:
            noise = rng.normal(0, 0.09)
            rr = np.exp(np.log(anchor) * rng.uniform(0.72, 1.08) + noise)
            se = rng.uniform(0.055, 0.14)
            lo = np.exp(np.log(rr) - 1.96 * se)
            hi = np.exp(np.log(rr) + 1.96 * se)
            rows.append({"protein": p, "cohort": c, "RR": rr, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)

with st.sidebar:
    st.markdown("## 🧬 Model controls")
    n = st.slider("Synthetic cohort size", 1000, 25000, 7000, step=500)
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=2026, step=1)
    pm_scale = st.slider("Population PM2.5 exposure multiplier", 0.5, 2.0, 1.0, step=0.05)
    case_multiplier = st.slider("Synthetic event-rate multiplier", 0.5, 4.0, 1.0, step=0.1)
    risk_cut = st.slider("High-risk threshold", 0.01, 0.25, 0.06, step=0.005)
    st.markdown("---")
    st.markdown("### 14-protein signature")
    for group, ps in PROTEIN_GROUPS.items():
        st.caption(group)
        st.markdown(" ".join([f"<span class='pill'>{p}</span>" for p in ps]), unsafe_allow_html=True)

df = simulate_cohort(n=n, seed=seed, pm_scale=pm_scale, case_multiplier=case_multiplier)
model, features, X_train, X_test, y_train, y_test, pred, auc, ap, brier = train_model(df, n, seed, pm_scale, case_multiplier)
df["predicted_risk"] = model.predict_proba(df[features])[:, 1]

st.title("🫁 Lung Prevention Signal Studio")
st.markdown("""
<div class='glass'>
A colourful synthetic-data studio inspired by a plasma-proteomics lung cancer prevention study. It links a 14-protein plasma signature, clinical factors, particulate matter, EGFR-mutant clone pressure, KAC-like alveolar transitional biology and IL-1β-directed prevention-trial enrichment.
<br><br><span class='small-note'>Portfolio demonstration only. The data and predictions are synthetic and must not be used for diagnosis, screening, or treatment decisions.</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Synthetic cohort", f"{len(df):,}")
k2.metric("Incident cases", f"{int(df.future_lung_cancer.sum()):,}", f"{df.future_lung_cancer.mean()*100:.2f}%")
k3.metric("Median PM2.5", f"{df.PM2_5.median():.1f}")
k4.metric("Model ROC-AUC", f"{auc:.3f}")
k5.metric("Average precision", f"{ap:.3f}")
k6.metric("Brier score", f"{brier:.3f}")

tabs = st.tabs([
    "🌈 Cohort atlas", "📈 Model performance", "🧪 Protein biology", "🧫 Single-cell KAC atlas",
    "⏳ Longitudinal signals", "🌫️ PM/IL-1β simulator", "🎯 Prevention trial designer", "🌍 External validation",
    "👤 Patient simulator", "📦 Export"
])

with tabs[0]:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = px.scatter(df.sample(min(len(df), 4500), random_state=1), x="PM2_5", y="signature_score", color="risk_tier", size="predicted_risk", hover_data=["age", "sex", "smoking_status", "pack_years", "COPD", "KAC_index", "predicted_risk"], title="Tumour-promotion plasma signature rises with environmental and inflammatory burden", color_discrete_sequence=COLORWAY)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=540, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.violin(df, x="smoking_status", y="signature_score", color="smoking_status", box=True, points=False, title="Signature distribution by smoking status", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=250, showlegend=False, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
        fig = px.histogram(df, x="predicted_risk", color="future_lung_cancer", nbins=65, marginal="box", title="Predicted risk distribution", color_discrete_sequence=["#33F2A0", "#FF4B4B"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=270, bargap=0.04, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    st.markdown("### Risk-tier summary")
    summary = df.groupby("risk_tier", observed=True).agg(n=("future_lung_cancer", "size"), observed_incidence=("future_lung_cancer", "mean"), median_predicted_risk=("predicted_risk", "median"), median_signature=("signature_score", "median"), median_KAC=("KAC_index", "median"), median_PM25=("PM2_5", "median"), current_smoker_fraction=("smoke_code", lambda x: np.mean(x == 2))).reset_index()
    st.dataframe(summary.style.format({"observed_incidence": "{:.2%}", "median_predicted_risk": "{:.2%}", "median_signature": "{:.2f}", "median_KAC": "{:.2f}", "median_PM25": "{:.1f}", "current_smoker_fraction": "{:.1%}"}), width="stretch")

with tabs[1]:
    c1, c2 = st.columns(2)
    fpr, tpr, _ = roc_curve(y_test, pred)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Gradient boosting AUC={auc:.3f}", line=dict(width=4, color="#00D4FF")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#BBBBBB")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", template=PLOTLY_TEMPLATE, height=420, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with c2:
        prec, rec, _ = precision_recall_curve(y_test, pred)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP={ap:.3f}", line=dict(width=4, color="#FF4ECD")))
        fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", template=PLOTLY_TEMPLATE, height=420, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        prob_true, prob_pred = calibration_curve(y_test, pred, n_bins=8, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="markers+lines", name="Calibration", marker=dict(size=10, color="#33F2A0")))
        mx = max(float(np.max(prob_pred)), float(np.max(prob_true)), 0.01)
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Ideal", line=dict(dash="dash")))
        fig.update_layout(title="Calibration", xaxis_title="Mean predicted risk", yaxis_title="Observed incidence", template=PLOTLY_TEMPLATE, height=420, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with c4:
        perm = permutation_importance(model, X_test, y_test, n_repeats=6, random_state=42, scoring="roc_auc")
        imp = pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(20)
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Permutation importance", color="importance", color_continuous_scale=CONTINUOUS_SCALE)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, yaxis={"categoryorder":"total ascending"}, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with st.expander("Compare model families"):
        small_features = ["age", "smoke_code", "pack_years", "COPD", "PM2_5", "IL1B_axis", "EGFR_clone_proxy", "KAC_index", "signature_score"]
        Xs = df[small_features]
        ys = df["future_lung_cancer"]
        models = {"Logistic regression": LogisticRegression(max_iter=1000), "Random forest": RandomForestClassifier(n_estimators=140, random_state=3, min_samples_leaf=8), "Gradient boosting": GradientBoostingClassifier(random_state=4, n_estimators=140, learning_rate=0.04)}
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=4)
        rows = []
        for name, clf in models.items():
            scores = cross_val_score(clf, Xs, ys, scoring="roc_auc", cv=cv)
            rows.append({"model": name, "mean_AUC": scores.mean(), "sd_AUC": scores.std()})
        st.dataframe(pd.DataFrame(rows).style.format({"mean_AUC":"{:.3f}", "sd_AUC":"{:.3f}"}), width="stretch")

with tabs[2]:
    st.markdown("### Protein signature heatmap")
    ordered = df.sort_values("signature_score").sample(min(len(df), 700), random_state=4).sort_values("signature_score")
    fig = px.imshow(ordered[PROTEINS].T, aspect="auto", color_continuous_scale=CONTINUOUS_SCALE, title="NPX-like expression across synthetic individuals, sorted by 14-protein score", labels=dict(x="Individuals", y="Protein", color="NPX"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor=PAPER_BG)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        rr = pd.DataFrame({"protein": list(RR_ANCHOR.keys()), "relative_risk_anchor": list(RR_ANCHOR.values())}).sort_values("relative_risk_anchor")
        fig = px.bar(rr, x="relative_risk_anchor", y="protein", orientation="h", color="relative_risk_anchor", color_continuous_scale=CONTINUOUS_SCALE, title="Relative-risk anchors used to shape synthetic data")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with c2:
        corr = df[PROTEINS + ["signature_score", "KAC_index", "IL1B_axis", "PM2_5"]].corr().loc[PROTEINS, ["signature_score", "KAC_index", "IL1B_axis", "PM2_5"]]
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Protein correlation with signature, KAC, inflammation and PM2.5")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor=PAPER_BG)
        st.plotly_chart(fig, width="stretch")

with tabs[3]:
    sc = simulate_single_cell(seed=seed)
    c1, c2 = st.columns([1.2, .8])
    with c1:
        color_by = st.radio("Colour cells by", ["cell_type", "signature_score", "KAC_score"], horizontal=True)
        fig = px.scatter(sc, x="UMAP1", y="UMAP2", color=color_by, hover_data=["cell_type", "signature_score", "KAC_score"], title="Synthetic single-cell atlas, epithelial, myeloid and KAC-like states", color_discrete_sequence=COLORWAY, color_continuous_scale=CONTINUOUS_SCALE)
        fig.update_traces(marker=dict(size=5, opacity=0.75))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=620, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
    with c2:
        agg = sc.groupby("cell_type").agg(mean_signature=("signature_score", "mean"), mean_KAC=("KAC_score", "mean"), n=("cell_type", "size")).reset_index().sort_values("mean_signature")
        fig = px.bar(agg, x="mean_signature", y="cell_type", orientation="h", color="mean_KAC", color_continuous_scale=CONTINUOUS_SCALE, title="Cell-type enrichment of synthetic 14-protein programme")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=440, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(agg.style.format({"mean_signature":"{:.2f}", "mean_KAC":"{:.2f}"}), width="stretch")

with tabs[4]:
    tc = simulate_timecourses(seed=seed)
    fig = px.line(tc.groupby(["case_status", "months_to_diagnosis"], as_index=False)["signature_score"].mean(), x="months_to_diagnosis", y="signature_score", color="case_status", markers=True, title="Synthetic longitudinal plasma-signature rise before diagnosis", color_discrete_sequence=["#FF4B4B", "#33F2A0"])
    fig.update_layout(template=PLOTLY_TEMPLATE, height=520, xaxis_autorange="reversed", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
    st.plotly_chart(fig, width="stretch")
    heat = tc.pivot_table(index="person", columns="months_to_diagnosis", values="signature_score")
    fig = px.imshow(heat, aspect="auto", color_continuous_scale=CONTINUOUS_SCALE, title="Longitudinal heatmap, individuals x months before diagnosis")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor=PAPER_BG)
    st.plotly_chart(fig, width="stretch")

with tabs[5]:
    st.markdown("### Scenario engine for tumour-promotion pressure")
    c1, c2, c3 = st.columns(3)
    pm_delta = c1.slider("Extra PM2.5 challenge", 0.0, 25.0, 8.0, .5)
    il1b_delta = c2.slider("Extra IL-1β challenge", 0.0, 3.0, 0.9, .1)
    blocker = c3.slider("IL-1β blockade strength", 0.0, 1.0, 0.45, .05)
    base = df.sample(min(len(df), 2500), random_state=2).copy()
    scenarios = []
    for label, dpm, dil, block in [("Baseline", 0, 0, 0), ("PM challenge", pm_delta, 0, 0), ("PM + IL-1β", pm_delta, il1b_delta, 0), ("PM + IL-1β + blockade", pm_delta, il1b_delta, blocker)]:
        tmp = base.copy()
        tmp["PM2_5"] = tmp["PM2_5"] + dpm
        tmp["IL1B_axis"] = tmp["IL1B_axis"] + dil * (1 - block)
        tmp["signature_score"] = tmp["signature_score"] + 0.035 * dpm + 0.55 * dil * (1 - block)
        tmp["KAC_index"] = expit(np.log(tmp["KAC_index"]/(1-tmp["KAC_index"].clip(.001,.999))) + 0.045*dpm + 0.72*dil*(1-block))
        tmp["scenario"] = label
        scenarios.append(tmp)
    scen = pd.concat(scenarios)
    fig = px.box(scen, x="scenario", y="signature_score", color="scenario", title="Environmental and inflammatory challenges shift the plasma signature", color_discrete_sequence=COLORWAY)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=430, showlegend=False, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
    st.plotly_chart(fig, width="stretch")
    fig = px.violin(scen, x="scenario", y="KAC_index", color="scenario", box=True, title="KAC-like transitional-state expansion under scenario pressure", color_discrete_sequence=COLORWAY)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=430, showlegend=False, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
    st.plotly_chart(fig, width="stretch")

with tabs[6]:
    st.markdown("### Prevention trial enrichment concept")
    trial_n = st.slider("Trial size", 500, 50000, 10000, 500)
    rrr_low = st.slider("Relative risk reduction in low-signature group", 0.0, 0.5, 0.08, 0.01)
    rrr_high = st.slider("Relative risk reduction in high-signature group", 0.0, 0.8, 0.42, 0.01)
    df2 = df.copy()
    df2["screen_positive"] = df2["predicted_risk"] >= risk_cut
    screen_n = int(df2.screen_positive.sum())
    screen_cases = int(df2.loc[df2.screen_positive, "future_lung_cancer"].sum())
    enriched_inc = df2.loc[df2.screen_positive, "future_lung_cancer"].mean() if screen_n else 0
    background_inc = df2["future_lung_cancer"].mean()
    response_mean = df2.loc[df2.screen_positive, "anti_IL1B_response_index"].mean() if screen_n else 0
    relative_risk_reduction = rrr_low + (rrr_high - rrr_low) * response_mean
    absolute_risk_reduction = enriched_inc * relative_risk_reduction
    nnt = 1 / absolute_risk_reduction if absolute_risk_reduction > 0 else np.inf
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Screen-positive", f"{screen_n:,}", f"{screen_n/len(df2):.1%}")
    c2.metric("Cases captured", f"{screen_cases:,}", f"{screen_cases/max(df2.future_lung_cancer.sum(),1):.1%}")
    c3.metric("Enriched incidence", f"{enriched_inc:.2%}", f"{enriched_inc/background_inc:.1f}× background" if background_inc else "")
    c4.metric("Conceptual NNT", f"{nnt:.0f}" if np.isfinite(nnt) else "—", f"RRR {relative_risk_reduction:.0%}")
    c5.metric("Expected prevented", f"{trial_n * absolute_risk_reduction:,.0f}")
    qs = np.linspace(.50, .99, 30)
    rows = []
    for q in qs:
        cut = df2.predicted_risk.quantile(q)
        selected = df2[df2.predicted_risk >= cut]
        inc = selected.future_lung_cancer.mean()
        resp = selected.anti_IL1B_response_index.mean()
        rrr = rrr_low + (rrr_high - rrr_low) * resp
        arr = inc * rrr
        rows.append({"selected_fraction": 1-q, "threshold": cut, "incidence": inc, "NNT": 1/arr if arr > 0 else np.nan, "response_index": resp})
    curve = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["selected_fraction"],
            y=curve["NNT"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(
                size=10,
                color=curve["response_index"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Response index"),
            ),
            text=[f"Threshold={t:.3f}<br>Incidence={i:.2%}<br>Response={r:.2f}" for t, i, r in zip(curve["threshold"], curve["incidence"], curve["response_index"])],
            hovertemplate="Selected fraction=%{x:.1%}<br>NNT=%{y:.1f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Selecting a higher-risk subgroup lowers conceptual NNT", template=PLOTLY_TEMPLATE, height=500, xaxis_tickformat=".0%", xaxis_title="Selected cohort fraction", yaxis_title="Conceptual NNT", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
    st.plotly_chart(fig, width="stretch")

with tabs[7]:
    ev = external_validation(seed=seed)
    selected_protein = st.selectbox("Protein", PROTEINS, index=0)
    sub = ev[ev.protein == selected_protein].sort_values("RR")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["RR"], y=sub["cohort"], mode="markers", marker=dict(size=13, color="#FF4ECD"), error_x=dict(type="data", symmetric=False, array=sub["hi"]-sub["RR"], arrayminus=sub["RR"]-sub["lo"])))
    fig.add_vline(x=1, line_dash="dash", line_color="white")
    fig.update_layout(title=f"Synthetic external-validation forest plot, {selected_protein}", xaxis_title="Relative risk", yaxis_title="Cohort", template=PLOTLY_TEMPLATE, height=520, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
    st.plotly_chart(fig, width="stretch")
    matrix = ev.pivot_table(index="protein", columns="cohort", values="RR")
    fig = px.imshow(matrix, color_continuous_scale=CONTINUOUS_SCALE, title="Relative-risk matrix across synthetic cohorts")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=620, paper_bgcolor=PAPER_BG)
    st.plotly_chart(fig, width="stretch")

with tabs[8]:
    st.markdown("### Interactive single-person simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 40, 85, 62)
        smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"], index=1)
        pack = st.slider("Pack years", 0.0, 90.0, 18.0, 1.0)
    with c2:
        copd = st.toggle("COPD history", value=False)
        pm25 = st.slider("PM2.5 exposure", 3.0, 38.0, 12.0, 0.5)
        il1b = st.slider("IL-1β-like inflammatory axis", -1.5, 3.5, 0.4, 0.1)
    with c3:
        egfr = st.toggle("EGFR-mutant clone proxy", value=False)
        kac_shift = st.slider("KAC-like state strength", 0.0, 1.0, 0.35, 0.05)
        protein_boost = st.slider("Overall protein-signature elevation", -1.5, 3.0, 0.5, 0.1)
    smoke_code = {"Never":0, "Former":1, "Current":2}[smoking]
    row = {"age": age, "smoke_code": smoke_code, "pack_years": pack, "COPD": int(copd), "PM2_5": pm25, "IL1B_axis": il1b, "EGFR_clone_proxy": int(egfr), "KAC_index": kac_shift}
    med = df[PROTEINS].median()
    for p in PROTEINS:
        row[p] = float(med[p] + protein_boost * np.log(RR_ANCHOR[p]))
    Xnew = pd.DataFrame([row])[features]
    r = float(model.predict_proba(Xnew)[:, 1][0])
    tier = "Very high" if r >= .12 else "High" if r >= .06 else "Intermediate" if r >= .025 else "Low"
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk", f"{r:.2%}")
    c2.metric("Risk tier", tier)
    c3.metric("Response index", f"{float(expit(-0.4 + 1.35 * protein_boost + 0.55 * il1b)):.2f}")
    gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=r*100, delta={"reference": df.predicted_risk.median()*100}, gauge={"axis": {"range": [0, max(25, df.predicted_risk.quantile(.995)*100)]}, "bar": {"color": "#FF4ECD"}, "steps": [{"range": [0, 2.5], "color": "rgba(51,242,160,0.35)"}, {"range": [2.5, 6], "color": "rgba(255,176,0,0.35)"}, {"range": [6, 12], "color": "rgba(255,78,205,0.35)"}, {"range": [12, 25], "color": "rgba(255,75,75,0.35)"}]}))
    gauge.update_layout(title="Synthetic risk gauge", template=PLOTLY_TEMPLATE, height=360, paper_bgcolor=PAPER_BG)
    st.plotly_chart(gauge, width="stretch")
    radar_df = pd.DataFrame({"protein": PROTEINS, "value": [row[p] for p in PROTEINS], "group": [next(g for g, ps in PROTEIN_GROUPS.items() if p in ps) for p in PROTEINS]})
    fig = px.line_polar(radar_df, r="value", theta="protein", color="group", line_close=True, title="Synthetic plasma protein profile", color_discrete_sequence=COLORWAY)
    fig.update_traces(fill="toself")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=520, paper_bgcolor=PAPER_BG)
    st.plotly_chart(fig, width="stretch")

with tabs[9]:
    st.markdown("### Export synthetic data and model summary")
    sample = df.sample(min(len(df), 5000), random_state=11)
    csv = sample.to_csv(index=False).encode("utf-8")
    st.download_button("Download synthetic cohort CSV", csv, "synthetic_lung_prevention_cohort.csv", "text/csv")
    report = f"""# Lung Prevention Signal Studio synthetic report\n\nCohort size: {len(df):,}\nIncident cases: {int(df.future_lung_cancer.sum()):,} ({df.future_lung_cancer.mean():.2%})\nModel ROC-AUC: {auc:.3f}\nAverage precision: {ap:.3f}\nBrier score: {brier:.3f}\nHigh-risk threshold: {risk_cut:.3f}\n\nModules included: cohort atlas, model performance, protein biology, single-cell KAC atlas, longitudinal signals, PM/IL-1β simulator, prevention trial designer, external validation, patient simulator and data export.\n\nThis is synthetic demonstration output only.\n"""
    st.download_button("Download markdown report", report.encode("utf-8"), "lung_prevention_signal_report.md", "text/markdown")
    st.dataframe(sample.head(200), width="stretch")

st.markdown("---")
st.markdown("""
<div class='glass small-note'>
Scientific basis captured in this app: 14-protein plasma risk signature, clinical covariates, particulate-matter exposure, EGFR-driven early LUAD biology, KAC-like alveolar transitional states, longitudinal pre-diagnostic signal emergence and IL-1β-linked prevention stratification. The implementation uses synthetic data and simplified models for demonstration.
</div>
""", unsafe_allow_html=True)
