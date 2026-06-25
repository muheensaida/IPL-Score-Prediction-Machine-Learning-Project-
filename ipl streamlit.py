# streamlit_ipl_app.py

import math
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------------------------
# 1) PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title='IPL Score Predictor',
    layout="wide",
)
# Stylish translucent sidebar CSS
st.markdown("""
<style>
/* Sidebar background: translucent glassmorphism effect */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Sidebar radio labels */
[data-testid="stSidebar"] .stRadio > label > div {
    color: #f0f0f0 !important;
    font-weight: 500;
}

/* Active radio button highlight */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    transition: 0.3s ease;
}

/* Sidebar section title */
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
    text-shadow: 0px 0px 8px rgba(255,255,255,0.3);
}

/* Active radio item indicator (custom circle border) */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    margin: 3px 0px;
    padding: 5px 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border: 1px solid rgba(255, 255, 255, 0.4);
}

/* Glow effect on selected radio item */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] div[aria-checked="true"] {
    box-shadow: 0px 0px 10px 2px rgba(255, 0, 100, 0.5);
}

/* Sidebar overall text color */
[data-testid="stSidebar"] * {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)
# Optional button styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(90deg, rgba(255,0,100,0.9), rgba(255,80,80,0.7));
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, rgba(255,80,80,0.9), rgba(255,0,100,0.9));
    transform: scale(1.03);
    box-shadow: 0px 0px 10px rgba(255,0,100,0.5);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 2) OPTIONAL BACKGROUND + TITLE
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    /* nicer sidebar radio */
    div[data-baseweb="radio"] > div {
        gap: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: white;'>IPL Score Predictor</h1>", unsafe_allow_html=True)

# -------------------------------------------------
# 3) TRY TO LOAD DATASET & MODEL
# -------------------------------------------------
DATA_PATH = Path("ipl_data.csv")
MODEL_PATH = Path("ml_model.pkl")

df_raw = None
if DATA_PATH.exists():
    try:
        df_raw = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")

model = None
if MODEL_PATH.exists():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# -------------------------------------------------
# 4) SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.markdown("### 📍 Navigate to Section")
page = st.sidebar.radio(
    "",
    (
        "🏠 Project Overview",
        "📊 Dataset Overview",
        "📈 Interactive EDA",
        "⚙️ Model Performance",
        "🎯 Manual Prediction",
    ),
    index=0,
)

# Common teams — same 8 you used in notebook
TEAMS = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

def team_one_hot(team_name: str):
    """Return 8-length one-hot in fixed order of TEAMS"""
    return [1 if t == team_name else 0 for t in TEAMS]

# -------------------------------------------------
# 5) PAGE 1: PROJECT OVERVIEW
# -------------------------------------------------
def new_func():
    with st.expander("Description"):
        st.info("""A Simple ML Model to predict IPL Scores between teams in an ongoing match.
    To make sure the model results are accurate and reliable, the minimum number of overs considered is greater than 5 overs.""")

if page == "🏠 Project Overview":
    st.header("Objective 🎯")
    st.markdown(
        """
        Predict **first-innings IPL score** using current match situation:
        - Batting team  
        - Bowling team  
        - Runs, Wickets, Overs  
        - Last 5 overs performance
        """
    )

    st.subheader("Pipeline / Steps")
    st.markdown(
        """
        1. Data Preprocessing & Cleaning  
        2. Keep only 8 consistent IPL teams  
        3. Remove first 5 overs of every innings  
        4. One-hot encode **batting** and **bowling** teams  
        5. Train Regression Models (Decision Tree, Random Forest, XGB, etc.)  
        6. Choose **Random Forest** as final model  
        7. Expose model via **Streamlit UI**  
        """
    )

    st.subheader("Features Used")
    st.markdown(
        """
        - 8 one-hot features for **batting team**  
        - 8 one-hot features for **bowling team**  
        - `runs`, `wickets`, `overs`, `runs_last_5`, `wickets_last_5`  
        """
    )

    st.info("Go to **🎯 Manual Prediction** from the sidebar to get score prediction for a live / test situation.")

# -------------------------------------------------
# 6) PAGE 2: DATASET OVERVIEW
# -------------------------------------------------
elif page == "📊 Dataset Overview":
    st.header("Dataset Overview 📊")

    if df_raw is None:
        st.warning("`ipl_data.csv` not found in this folder. Put the dataset in the same folder as this app.")
    else:
        st.success(f"Dataset successfully loaded with shape: {df_raw.shape}")
        st.write("**First 10 rows:**")
        st.dataframe(df_raw.head(10))

        st.write("**Column info:**")
        st.write(df_raw.dtypes)

        st.write("**Number of unique values per column:**")
        st.write(df_raw.nunique())

        st.write("**Consistent teams in data (from notebook):**")
        const_teams = [
            'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
            'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
            'Delhi Daredevils', 'Sunrisers Hyderabad'
        ]
        st.write(const_teams)

        st.caption("These are the same 8 teams you used while training — we filter others out before training.")

# -------------------------------------------------
# 7) PAGE 3: INTERACTIVE EDA
# -------------------------------------------------
elif page == "📈 Interactive EDA":
    st.header("Interactive EDA 📈")

    if df_raw is None:
        st.warning("No dataset found. Please place `ipl_data.csv` next to this app.")
    else:
        # make a cleaned copy like notebook (drop irrelevant, filter teams, overs>=5)
        df = df_raw.copy()

        # drop irrelevant columns (if present)
        irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
        df = df.drop([c for c in irrelevant if c in df.columns], axis=1)

        # filter to consistent teams
        df = df[(df['bat_team'].isin(TEAMS)) & (df['bowl_team'].isin(TEAMS))]

        # remove first 5 overs
        df = df[df['overs'] >= 5.0]

        st.markdown("#### 1. Runs Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['total'], bins=10, kde=False, ax=ax1)
        ax1.set_title("Runs (Total) Distribution")
        st.pyplot(fig1)

        st.markdown("#### 2. Overs vs Total (sample 500 rows)")
        st.caption("Higher overs obviously means closer to innings end → higher totals.")
        sample_df = df.sample(min(500, len(df)), random_state=42)
        st.scatter_chart(sample_df[['overs', 'total']].rename(columns={'overs': 'Overs', 'total': 'Total'}))# ---- Clearer Overs vs Total visualization ----

        st.markdown("#### 3. Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", ax=ax2)
        ax2.set_title("Correlation Heatmap")
        st.pyplot(fig2)

        st.markdown("#### 4. Batting Team counts")
        st.bar_chart(df['bat_team'].value_counts())

# -------------------------------------------------
# 8) PAGE 4: MODEL PERFORMANCE
# -------------------------------------------------
elif page == "⚙️ Model Performance":
    st.header("Model Performance ⚙️")

    if model is None:
        st.warning("Model file `ml_model.pkl` not found. Train and pickle it first.")
    elif df_raw is None:
        st.warning("Dataset `ipl_data.csv` not found. Cannot recompute metrics.")
    else:
        st.write("We will do a **quick re-evaluation** of the saved model on the original CSV.")

        # replicate training-time preprocessing (important!)
        df = df_raw.copy()
        # drop irrelevant
        irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
        df = df.drop([c for c in irrelevant if c in df.columns], axis=1)
        # keep 8 consistent teams
        df = df[(df['bat_team'].isin(TEAMS)) & (df['bowl_team'].isin(TEAMS))]
        # overs >= 5.0
        df = df[df['overs'] >= 5.0]

        # one-hot EXACTLY like training (bat_team then bowl_team)
        # we will build it manually to guarantee order
        rows = []
        targets = []
        for _, row in df.iterrows():
            bt = row['bat_team']
            bw = row['bowl_team']
            runs = row['runs']
            wickets = row['wickets']
            overs = row['overs']
            runs_last_5 = row['runs_last_5']
            wickets_last_5 = row['wickets_last_5']
            total = row['total']

            vec = []
            vec += team_one_hot(bt)
            vec += team_one_hot(bw)
            vec += [runs, wickets, overs, runs_last_5, wickets_last_5]

            rows.append(vec)
            targets.append(total)

        X = np.array(rows)
        y = np.array(targets)

        # predict
        y_pred = model.predict(X)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y, y_pred)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("MAE", f"{mae:.2f}")
        colB.metric("RMSE", f"{rmse:.2f}")
        colC.metric("MSE", f"{mse:.2f}")
        colD.metric("R²", f"{r2:.3f}")

        st.caption("These metrics are recomputed inside the app using the same CSV and the saved Random Forest model.")

# -------------------------------------------------
# 9) PAGE 5: MANUAL PREDICTION (YOUR ORIGINAL FORM)
# -------------------------------------------------
elif page == "🎯 Manual Prediction":
    new_func()

    st.header("Manual Prediction 🎯")
    st.info("Enter current match situation (>= 5 overs) and get the predicted final score.")

    # Form layout
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Select the Batting Team", TEAMS)
    with col2:
        bowling_team = st.selectbox("Select the Bowling Team", TEAMS)

    if batting_team == bowling_team:
        st.error("Batting and Bowling team must be different.")

    col3, col4 = st.columns(2)
    with col3:
        overs = st.number_input("Enter Current Over", min_value=5.0, max_value=19.5, value=5.1, step=0.1, format="%.1f")
        if overs - math.floor(overs) > 0.5:
            st.error("Please enter a valid over (only .0 to .5 allowed)")
    with col4:
        runs = st.number_input("Enter Current Runs", min_value=0, max_value=400, value=0, step=1)

    wickets = st.slider("Wickets fallen till now", 0, 9, 0)

    col5, col6 = st.columns(2)
    with col5:
        runs_in_prev_5 = st.number_input("Runs in last 5 overs", min_value=0, max_value=400, value=0, step=1)
    with col6:
        wickets_in_prev_5 = st.number_input("Wickets in last 5 overs", min_value=0, max_value=9, value=0, step=1)

    # Build feature vector in the SAME order as training
    features = []
    features += team_one_hot(batting_team)
    features += team_one_hot(bowling_team)
    features += [runs, int(wickets), overs, runs_in_prev_5, wickets_in_prev_5]
    features = np.array([features])

    if st.button("Predict Score"):
        if model is None:
            st.error("Model not loaded. Make sure 'ml_model.pkl' is in the same folder.")
        elif batting_team == bowling_team:
            st.error("Choose two different teams.")
        else:
            pred = model.predict(features)
            final_score = int(round(pred[0]))
            # st.success(f"PREDICTED MATCH SCORE : {final_score}"
            st.success(f'PREDICTED MATCH SCORE : {final_score-5} to {final_score+5}')


    st.caption("Tip: this follows the exact notebook logic you used (8+8 team one-hots + 5 numeric features).")
