import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq
import gdown

# ========================
# Config
# ========================
st.set_page_config(page_title="Signalen + GPS Viewer", layout="wide")

DATA_URL = st.secrets.get("DATA_URL", None)  # optioneel: gdrive url/id via secrets
DATA_FILE = "data.parquet"                  # lokale cache naam

# Kolomnamen in je dataset (pas aan indien nodig)
TIME_COL = "Timestamp"
LAT_COL = "GPS_x"
LON_COL = "GPS_y"

# ========================
# Helpers
# ========================
@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()
    # Forceer datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

def ensure_data_file() -> str:
    """Zorgt dat DATA_FILE lokaal bestaat (download indien nodig)."""
    if os.path.exists(DATA_FILE):
        return DATA_FILE

    if DATA_URL:
        # DATA_URL kan een volledige Google Drive link zijn of een file id
        try:
            gdown.download(DATA_URL, DATA_FILE, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Download mislukt: {e}")
            st.stop()
        return DATA_FILE

    st.error(
        "Geen data gevonden. Zet DATA_URL in Streamlit secrets of plaats data.parquet naast deze app."
    )
    st.stop()

def downsample_ordered(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Eenvoudige downsample: neem elke n-de rij om max_points te benaderen."""
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()

# ========================
# Load data
# ========================
path = ensure_data_file()
df = load_parquet(path)

# ========================
# UI: selectie signalen
# ========================
st.title("Signalen + GPS Viewer")

numeric_cols = [c for c in df.columns if c not in [TIME_COL, LAT_COL, LON_COL]]
if len(numeric_cols) < 3:
    st.warning("Er zijn minder dan 3 signaal-kolommen beschikbaar.")
    st.stop()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Instellingen")

    sig1 = st.selectbox("Signaal 1", numeric_cols, index=0)
    sig2 = st.selectbox("Signaal 2", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    sig3 = st.selectbox("Signaal 3", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)

    if len({sig1, sig2, sig3}) < 3:
        st.warning("Kies drie verschillende signalen.")
        st.stop()

    # Tijdvenster kiezen
    min_ts = df[TIME_COL].min().to_pydatetime()
    max_ts = df[TIME_COL].max().to_pydatetime()

    start_dt = st.date_input("Start datum", value=min_ts.date(), min_value=min_ts.date(), max_value=max_ts.date())
    start_time = st.time_input("Start tijd", value=min_ts.time())

    end_dt = st.date_input("Eind datum", value=max_ts.date(), min_value=min_ts.date(), max_value=max_ts.date())
    end_time = st.time_input("Eind tijd", value=max_ts.time())

    start = pd.Timestamp.combine(pd.Timestamp(start_dt), pd.Timestamp(start_time).time())
    end = pd.Timestamp.combine(pd.Timestamp(end_dt), pd.Timestamp(end_time).time())

    if start >= end:
        st.warning("Start moet voor eind liggen.")
        st.stop()

    # GPS smoothing / venster
    smooth_window_s = st.slider("Smoothing venster (seconden)", min_value=0, max_value=60, value=0, step=1)
    max_points = st.slider("Max punten in grafiek", min_value=500, max_value=50000, value=5000, step=500)

# Filter op tijdvenster
sub = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
if len(sub) < 2:
    st.warning("Te weinig data in dit tijdvenster.")
    st.stop()

# =======================
# Export geselecteerd tijdvenster
# =======================
# Exporteert de *volledige* selectie (niet-downsampled) naar CSV
export_cols = ["Timestamp", LON_COL, LAT_COL, sig1, sig2, sig3]
export_df = sub[export_cols].copy()

# Bestandsnaam met tijdvenster (zonder ':' zodat dit overal werkt)
start_str = pd.Timestamp(start_dt).strftime("%Y%m%d_%H%M%S")
end_str = pd.Timestamp(end_dt).strftime("%Y%m%d_%H%M%S")
csv_name = f"selectie_{start_str}_tot_{end_str}.csv"

csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download geselecteerd tijdvenster (CSV)",
    data=csv_bytes,
    file_name=csv_name,
    mime="text/csv",
)

# Downsample voor visualisatie
sub_ds = downsample_ordered(sub, max_points=max_points)

# Smoothing van GPS (optioneel)
if smooth_window_s > 0:
    # Benader aantal samples in window via mediane dt
    dt_seconds = sub_ds[TIME_COL].diff().dt.total_seconds().median()
    if pd.isna(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 1.0
    win = int(max(1, round(smooth_window_s / dt_seconds)))

    sub_ds[LAT_COL] = sub_ds[LAT_COL].rolling(win, center=True, min_periods=1).mean()
    sub_ds[LON_COL] = sub_ds[LON_COL].rolling(win, center=True, min_periods=1).mean()

# ========================
# Plot: signalen
# ========================
with col_right:
    st.subheader("Signalen in tijd")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=sub_ds[TIME_COL], y=sub_ds[sig1], mode="lines", name=sig1))
    fig.add_trace(go.Scatter(x=sub_ds[TIME_COL], y=sub_ds[sig2], mode="lines", name=sig2))
    fig.add_trace(go.Scatter(x=sub_ds[TIME_COL], y=sub_ds[sig3], mode="lines", name=sig3))

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tijd",
        yaxis_title="Waarde",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================
# Plot: GPS track
# ========================
st.subheader("GPS track (lat/lon)")
gps_fig = go.Figure()

gps_fig.add_trace(
    go.Scatter(
        x=sub_ds[LON_COL],
        y=sub_ds[LAT_COL],
        mode="lines+markers",
        marker=dict(size=4),
        name="Track",
        text=sub_ds[TIME_COL].astype(str),
        hovertemplate="Lon: %{x}<br>Lat: %{y}<br>Tijd: %{text}<extra></extra>",
    )
)

gps_fig.update_layout(
    height=450,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
)

st.plotly_chart(gps_fig, use_container_width=True)

# ========================
# Tabel preview
# ========================
with st.expander("Toon data (preview)"):
    st.dataframe(sub_ds[[TIME_COL, LON_COL, LAT_COL, sig1, sig2, sig3]].head(200))
