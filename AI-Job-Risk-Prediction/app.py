import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Risk Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .risk-low    { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 8px; }
    .risk-medium { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 8px; }
    .risk-high   { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 8px; }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH  = "final_dataset_mlhackathon.csv"
MODEL_PATH = "final_model_svm.pkl"

RISK_LABELS = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}
RISK_CSS    = {0: "risk-low",    1: "risk-medium",    2: "risk-high"}
RISK_TIPS   = {
    0: "Your job has low AI automation risk! Focus on deepening domain expertise and leadership skills.",
    1: "Moderate risk. Upskill in AI tools, Python, or cloud technologies to future-proof your career.",
    2: "High risk of AI disruption. Consider transitioning to AI-adjacent roles or acquiring new technical skills urgently."
}

CAT_COLS = ["job_title", "experience_level", "education_level", "primary_skill", "salary_bucket"]
NUM_COLS = ["skill_demand_score", "job_openings", "job_survival_class"]
TARGET   = "ai_risk_category"

JOB_TITLES       = ['Data Scientist', 'Software Engineer', 'Data Analyst', 'DevOps Engineer',
                     'Cybersecurity Analyst', 'Cloud Engineer', 'Business Analyst',
                     'AI Researcher', 'ML Engineer', 'Product Manager']
EXP_LEVELS       = ['Entry', 'Mid', 'Senior']
EDU_LEVELS       = ['Bachelor', 'Master', 'PhD']
PRIMARY_SKILLS   = ['Python', 'Java', 'SQL', 'Docker', 'Security', 'AWS', 'Excel',
                    'Deep Learning', 'Strategy']
SALARY_BUCKETS   = ['Low', 'Medium', 'High']

# ─── Load / Train Model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_or_train_model():
    """Try loading pkl; if fails, retrain a quick SVM from CSV."""
    # Try loading existing pkl
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model, "loaded"
        except Exception:
            pass

    # Fallback: retrain from CSV
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    X = df[CAT_COLS + NUM_COLS]
    y = df[TARGET]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline, "trained"


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return pd.read_csv(DATA_PATH)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Navigation")
    page = st.radio("Go to:", ["🏠 Home & Predict", "📊 EDA Dashboard", "📈 Model Info"])

    st.markdown("---")
    st.markdown("### 📂 Upload Files")

    uploaded_csv   = st.file_uploader("Dataset CSV", type=["csv"])
    uploaded_model = st.file_uploader("Model PKL",   type=["pkl"])

    if uploaded_csv:
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded_csv.read())
        st.success("Dataset uploaded!")
        load_data.clear()

    if uploaded_model:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.read())
        st.success("Model uploaded!")
        load_or_train_model.clear()

    st.markdown("---")
    st.markdown("**🏆 AI Hunger Games Hackathon**")
    st.markdown("*Innomatics Research Labs*")
    st.caption("Target: AI Risk Category Prediction")


# ─── Load Resources ────────────────────────────────────────────────────────────
df = load_data()
model, model_source = load_or_train_model()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home & Predict
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<p class="main-title">🤖 AI Job Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI Hunger Games Hackathon — Innomatics Research Labs</p>', unsafe_allow_html=True)
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📋 Total Records", f"{len(df):,}")
    with c2:
        st.metric("🔢 Features", len(CAT_COLS + NUM_COLS))
    with c3:
        low_pct = round(df[TARGET].value_counts(normalize=True).get(0, 0) * 100, 1)
        st.metric("🟢 Low Risk Jobs", f"{low_pct}%")
    with c4:
        hi_pct = round(df[TARGET].value_counts(normalize=True).get(2, 0) * 100, 1)
        st.metric("🔴 High Risk Jobs", f"{hi_pct}%")

    st.markdown("---")
    st.markdown("### 🔍 Predict Your AI Job Risk")
    st.markdown("Fill in the details below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        job_title        = st.selectbox("💼 Job Title",        JOB_TITLES)
        experience_level = st.selectbox("🎯 Experience Level", EXP_LEVELS)
        education_level  = st.selectbox("🎓 Education Level",  EDU_LEVELS)

    with col2:
        primary_skill      = st.selectbox("🛠️ Primary Skill",        PRIMARY_SKILLS)
        salary_bucket      = st.selectbox("💰 Salary Bucket",         SALARY_BUCKETS)
        job_survival_class = st.selectbox("🔮 Job Survival Class",    [0, 1, 2],
                                          format_func=lambda x: {0: "0 - Vulnerable", 1: "1 - Stable", 2: "2 - Growing"}[x])

    with col3:
        skill_demand_score = st.slider("📈 Skill Demand Score", min_value=60,   max_value=99,   value=79)
        job_openings       = st.slider("📋 Job Openings",       min_value=1000, max_value=50000, value=25000, step=500)

    st.markdown("")
    if st.button("🚀 Predict AI Risk Category"):
        input_data = pd.DataFrame([{
            "job_title": job_title,
            "experience_level": experience_level,
            "education_level": education_level,
            "primary_skill": primary_skill,
            "salary_bucket": salary_bucket,
            "skill_demand_score": skill_demand_score,
            "job_openings": job_openings,
            "job_survival_class": job_survival_class,
}])
        try:
            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            # ─── Show Exact Risk Percentages ───
            low_pct    = round(proba[0] * 100, 2)
            medium_pct = round(proba[1] * 100, 2)
            high_pct   = round(proba[2] * 100, 2)

            st.markdown("### 📊 Risk Percentage Breakdown")

            st.write(f"🟢 Low Risk: {low_pct}%")
            st.write(f"🟡 Medium Risk: {medium_pct}%")
            st.write(f"🔴 High Risk: {high_pct}%")
            
            risk_score = round(max(proba) * 100, 2)
            st.metric("🔥 AI Risk Score", f"{risk_score}%")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2 = st.columns([1, 2])
        with r1:
            css_class = RISK_CSS[pred]
            label     = RISK_LABELS[pred]
            tip       = RISK_TIPS[pred]
            st.markdown(f'<div class="{css_class}"><h2>{label}</h2><p>{tip}</p></div>', unsafe_allow_html=True)

        with r2:
            st.markdown("**Prediction Confidence:**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors  = ["#28a745", "#ffc107", "#dc3545"]
            bars    = ax.barh(["Low Risk", "Medium Risk", "High Risk"], proba, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for bar, val in zip(bars, proba):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va="center", fontsize=10, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("---")

    # Target distribution
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### AI Risk Category Distribution")
        vc = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#28a745", "#ffc107", "#dc3545"]
        ax.pie(vc.values, labels=["Low Risk", "Medium Risk", "High Risk"],
               colors=colors, autopct="%1.1f%%", startangle=140,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("AI Risk Category", fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("### Job Title vs Avg AI Risk Score")
        avg_risk = df.groupby("job_title")["ai_risk_score"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(avg_risk.index, avg_risk.values,
                       color=plt.cm.RdYlGn_r(avg_risk.values / avg_risk.max()))
        ax.set_xlabel("Avg AI Risk Score")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Avg Risk Score by Job Title", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Experience Level vs Risk Category")
        ct = pd.crosstab(df["experience_level"], df[TARGET], normalize="index") * 100
        ct.columns = ["Low", "Medium", "High"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ct.plot(kind="bar", ax=ax, color=["#28a745", "#ffc107", "#dc3545"],
                edgecolor="white", rot=0)
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Risk")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Risk % by Experience Level", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        st.markdown("### Primary Skill Demand Scores")
        skill_demand = df.groupby("primary_skill")["skill_demand_score"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(skill_demand.index, skill_demand.values,
               color=plt.cm.Blues(np.linspace(0.4, 0.9, len(skill_demand))))
        ax.set_xlabel("Primary Skill")
        ax.set_ylabel("Avg Skill Demand Score")
        plt.xticks(rotation=45, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Avg Skill Demand Score by Skill", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Raw data preview
    st.markdown("---")
    st.markdown("### 🗂️ Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Showing 20 of {len(df):,} records | {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("## 📈 Model Information")
    st.markdown("---")

    source_msg = "✅ Loaded from uploaded `final_model_svm.pkl`" if model_source == "loaded" else \
                 "⚠️ PKL version mismatch — model retrained fresh from dataset (SVM, RBF kernel)"
    st.info(source_msg)

    st.markdown("### 🔬 Model Architecture")
    st.code(str(model), language="text")

    st.markdown("---")
    st.markdown("### 📊 Evaluation on Test Set (20% holdout)")

    with st.spinner("Evaluating..."):
        tmp_df = df[CAT_COLS + NUM_COLS + [TARGET]].dropna()
        X = tmp_df[CAT_COLS + NUM_COLS]
        y = tmp_df[TARGET]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Accuracy", f"{acc:.2%}")
    m2.metric("📋 Test Samples", f"{len(y_test):,}")
    m3.metric("🗂️ Classes", "3 (Low / Medium / High)")

    # Classification Report
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=["Low Risk", "Medium Risk", "High Risk"],
                                   output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
                 use_container_width=True)

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### 🧩 Feature Engineering Summary")
    st.markdown("""
    | Feature | Type | Role |
    |---------|------|------|
    | `job_title` | Categorical | Encoded via OneHotEncoder |
    | `experience_level` | Categorical | Encoded via OneHotEncoder |
    | `education_level` | Categorical | Encoded via OneHotEncoder |
    | `primary_skill` | Categorical | Encoded via OneHotEncoder |
    | `salary_bucket` | Categorical | Encoded via OneHotEncoder |
    | `ai_risk_score` | Numeric | Scaled via StandardScaler |
    | `skill_demand_score` | Numeric | Scaled via StandardScaler |
    | `job_openings` | Numeric | Scaled via StandardScaler |
    | `job_survival_class` | Numeric | Scaled via StandardScaler |
    """)