import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.signal import butter, filtfilt
from scipy.stats import f_oneway
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import base64
import os
import statsmodels.api as sm
import sklearn
<<<<<<< HEAD
from streamlit.components.v1 import html




# --- BEFORE ANY UI ---
st.set_page_config(layout="wide", page_title="XFI 3D Mouse Dashboard")

# Kill page side padding and force the components iframe to be full width
st.markdown("""
<style>
/* kill page padding and max-width */
main .block-container { padding-left:0 !important; padding-right:0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { padding: 0 !important; }

/* make ANY html component wrapper + iframe full width (covers multiple versions) */
[data-testid="stHtml"] { width: 100% !important; }
[data-testid="stIFrame"] { width: 100% !important; }
[data-testid="stIFrame"] > iframe { width: 100% !important; display:block !important; }

/* fallback classnames for older releases */
.css-18ni7ap, .css-1d391kg { width: 100% !important; }

/* optional: remove top padding so the viewer hugs the top edge */
.main .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

=======
>>>>>>> 478f20e6135056cc82468c4990ba604a81800600



# -------------------------------
# Load 3D mouse model from disk
# -------------------------------
def load_mouse_model_base64():
    model_path = "C:/Users/konra/OneDrive/Desktop/Axiome/voxel_mouse.glb"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# -------------------------------
# Generate HTML for model-viewer
# -------------------------------
def get_model_viewer_html():
    model_data = load_mouse_model_base64()
    src = f"data:application/octet-stream;base64,{model_data}" if model_data else "https://modelviewer.dev/shared-assets/models/Astronaut.glb"

    return f"""
      <script>
        // Force this component's iframe to 100% width (works across Streamlit versions)
        (function() {{
          try {{
            const frame = window.frameElement;
            if (frame) {{
              frame.style.width = '100%';
              if (frame.parentElement) frame.parentElement.style.width = '100%';
            }}
          }} catch (e) {{}}
        }})();
      </script>

      <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
      <style>
        html, body, #wrap {{ height: 100%; margin: 0; }}
        #wrap {{ width: 100%; }}
        model-viewer {{ width: 100%; height: 100%; background-color: #f0f0f0; }}
      </style>
      <div id="wrap">
        <model-viewer src="{src}" alt="3D Maus"
          auto-rotate auto-rotate-delay="0" rotation-per-second="30deg"
          camera-controls exposure="1" shadow-intensity="1"
          camera-orbit="45deg 90deg 2m">
        </model-viewer>
      </div>
    """



# -------------------------------
# Simulate data (60 seconds)
# -------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    mice = ["M001", "M002", "M003"]
    seconds = list(range(1, 61))
    tissues = ["Leber", "Niere"]
    dose_levels = [1.0, 1.5, 2.0]

    rows = []
    for i, mouse in enumerate(mice):
        for sec in seconds:
            for tissue in tissues:
                dose = dose_levels[i]
                base = 0.4 + 0.2 * np.sin(sec / 10.0 + i)
                jod = np.round(np.clip(np.random.normal(loc=base, scale=0.05), 0.05, 0.8), 3)
                timestamp = datetime(2023, 1, 1) + timedelta(seconds=sec)
                rows.append([mouse, sec, tissue, dose, jod, timestamp])

    return pd.DataFrame(rows, columns=["TierID", "Time", "Gewebe", "Dosis", "Jod", "Timestamp"])


# -------------------------------
# k-means
# -------------------------------

def classify_jod_levels(df, n_clusters=3):
    df = df.copy()
    valid = df["Jod"].notna()
    
    # Reshape required by KMeans
    jod_values = df.loc[valid, "Jod"].values.reshape(-1, 1)
    
    # Fit KMeans

    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(jod_values)

    # Order clusters by mean Jod
    means = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(means)
    label_map = {sorted_indices[0]: "Low", sorted_indices[1]: "Medium", sorted_indices[2]: "High"}
    
    # Apply mapped labels
    classified = pd.Series(clusters).map(label_map)
    
    # Insert into DataFrame
    df.loc[valid, "Auto_classified"] = classified.values
    df["Auto_classified"] = df["Auto_classified"].astype("category")

    return df



# -------------------------------
# Data cleaning functions
# -------------------------------
def apply_bandpass(df):
    b, a = butter(2, [0.1, 0.49], btype="bandpass", fs=1.0)
    def filter_group(group):
        jod = group["Jod"].values
        if len(jod) < 3:
            group["Jod"] = np.nan
        else:
            try:
                group["Jod"] = filtfilt(b, a, jod)
            except:
                group["Jod"] = np.nan
        return group
    return df.groupby(["TierID", "Gewebe"], group_keys=False).apply(filter_group)

def remove_outliers(df):
    q1 = df["Jod"].quantile(0.25)
    q3 = df["Jod"].quantile(0.75)
    iqr = q3 - q1
    return df[(df["Jod"] >= q1 - 1.5 * iqr) & (df["Jod"] <= q3 + 1.5 * iqr)]

def impute_missing(df):
    return df.groupby(["TierID", "Gewebe"])["Jod"].transform(lambda x: x.fillna(x.mean()))

# -------------------------------
# Inference functions
# -------------------------------
def run_anova(df):
    grouped = df.groupby("Gewebe")["Jod"].apply(list)
    return f_oneway(*grouped)

def decompose_time_series(df):
    result = {}
    try:
        series = df[df["Gewebe"] == "Leber"].sort_values("Timestamp")["Jod"]
        series.index = pd.to_datetime(df[df["Gewebe"] == "Leber"].sort_values("Timestamp")["Timestamp"])
        decomposition = seasonal_decompose(series, model='additive', period=10)
        result = {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "resid": decomposition.resid
        }
    except Exception as e:
        st.error(f"Time series decomposition failed: {e}")
    return result

# -------------------------------
# Forecasting function
# -------------------------------

def forecast_model(df, features, model_type, forecast_horizon):
    df = df.copy()
    df = pd.get_dummies(df, columns=["Gewebe", "TierID"], drop_first=True)
    df = df.dropna()

    X = df[features]
    X = sm.add_constant(X)  # Add intercept term
    y = df["Jod"]

    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_type == "Linear Regression":
        model = sm.OLS(y_train, X_train).fit()
    else:
        st.warning("Nur Lineare Regression wird aktuell unterstützt (kein Random Forest).")
        model = sm.OLS(y_train, X_train).fit()

    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    forecast_input = sm.add_constant(X.tail(forecast_horizon))
    forecast_values = model.predict(forecast_input)

    # Get last rows from original DataFrame before one-hot encoding
    original_tail = df.tail(forecast_horizon).copy()

    # Prepare result
    result_df = original_tail.copy()
    result_df["Forecast"] = forecast_values

    # Ensure required columns are presentf
    if "TierID" in result_df.columns:
        result_df["TierID"] = result_df["TierID"].astype(str)
    if "Gewebe" in result_df.columns:
        result_df["Gewebe"] = result_df["Gewebe"].astype(str)

    return result_df, rmse



# -------------------------------
# Streamlit layout
# -------------------------------

st.title("XFI Jodverteilung am 3D Mausmodell")
html(get_model_viewer_html(), height=420, scrolling=False)

# Load data
df = generate_data()
df["TierID"] = df["TierID"].astype(str)
df["Gewebe"] = df["Gewebe"].astype(str)

# Sidebar controls
st.sidebar.header("⚙️ Datenoptionen")
clean_method = st.sidebar.selectbox("Datenbereinigung", ["Keine", "Bandpass-Filter", "Ausreißer entfernen", "Imputation"])
update_timestamps = st.sidebar.checkbox("Timestamps auf Jetzt aktualisieren")
apply = st.sidebar.button("Anwenden")

if apply:
    if update_timestamps:
        df["Timestamp"] = datetime.now() + pd.to_timedelta(df["Time"], unit="s")
    if clean_method == "Bandpass-Filter":
        df = apply_bandpass(df)
    elif clean_method == "Ausreißer entfernen":
        df = remove_outliers(df)
    elif clean_method == "Imputation":
        df["Jod"] = impute_missing(df)
    st.session_state["df"] = df
else:
    if "df" not in st.session_state:
        st.session_state["df"] = df

df = st.session_state["df"]
df = classify_jod_levels(df)

# Tabs
tab_titles = ["📄 Rohdaten", "📊 Statistik", "📈 Visualisierung", "📉 Forecast"]
tabs = st.tabs(tab_titles)

for i, tab in enumerate(tabs):
    with tab:
        if i == 0:
            st.subheader("Rohdaten")
            df["TierID"] = df["TierID"].astype(str)
            st.dataframe(df)

        elif i == 1:
            st.subheader("Deskriptive Statistik")
            st.write(df.describe(include="all"))

            st.subheader("Inferenzstatistik")
            anova_result = run_anova(df)
            st.write("**ANOVA p-Wert (Gewebe):**", round(anova_result.pvalue, 5))

            decomposition = decompose_time_series(df)
            if decomposition:
                st.line_chart(decomposition["trend"], height=200, use_container_width=True)
                st.line_chart(decomposition["seasonal"], height=200, use_container_width=True)

        elif i == 2:
            st.subheader("Histogramm der Jodverteilung")
            fig_hist = px.histogram(df, x="Jod", color="Gewebe", barmode="overlay", nbins=10)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Boxplot der Jodverteilung")
            fig_box = px.box(df, x="Gewebe", y="Jod", color="Gewebe", points="all")
            st.plotly_chart(fig_box, use_container_width=True)

            # 🔽 ADD THIS BLOCK HERE
            st.subheader("Jodklassifikation (automatisch, clusteringbasiert)")
            fig_class = px.scatter(df, x="Time", y="Jod", color="Auto_classified", facet_col="TierID",
                                labels={"Auto_classified": "Klassifikation"})
            st.plotly_chart(fig_class, use_container_width=True)
            # 🔼 END BLOCK

            st.subheader("Jodverteilung über Zeit")
            fig_ts = px.line(df, x="Time", y="Jod", color="Gewebe", markers=True, facet_col="TierID",
                            labels={"Time": "Sekunden", "Jod": "Jod"})
            st.plotly_chart(fig_ts, use_container_width=True)

        elif i == 3:
            st.subheader("ML Forecast der Jodkonzentration")
            all_features = [col for col in df.columns if col not in ["Jod", "Timestamp"] and df[col].dtype in [np.number, np.int64, np.float64]]
            selected_features = st.multiselect("Features für das Modell", all_features, default=["Time", "Dosis"])
            model_choice = st.selectbox("Modell wählen", ["Linear Regression", "Random Forest"])
            forecast_horizon = st.slider("Vorhersage-Horizont (letzte N Werte)", 1, 20, 5)

            if st.button("Vorhersage starten"):
                forecast_df, rmse = forecast_model(df, selected_features, model_choice, forecast_horizon)
                st.write(f"RMSE: {rmse:.4f}")
                st.dataframe(forecast_df[["Timestamp", "Jod", "Forecast"]])
                st.line_chart(forecast_df.set_index("Timestamp")[["Jod", "Forecast"]])