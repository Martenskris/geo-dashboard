import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq

# =======================
# Config
# =======================
FILE = "TotaleTimetable_date.parquet"

import os
import streamlit as st
import gdown

FILE = "TotaleTimetable_date.parquet"

# In Streamlit Cloud staat dit in Secrets
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "").strip()

def ensure_dataset():
    if os.path.exists(FILE):
        return

    if not GDRIVE_FILE_ID:
        st.error("Secret GDRIVE_FILE_ID ontbreekt in Streamlit Cloud → App settings → Secrets.")
        st.stop()

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download"

    with st.status("Dataset downloaden van Google Drive…", expanded=True) as s:
        st.write("Drive file id:", GDRIVE_FILE_ID)
        try:
            # fuzzy=True laat gdown ook share-links/id's slim interpreteren
            gdown.download(url, FILE, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Download faalde: {type(e).__name__}: {e}")
            st.stop()
        s.update(label="Dataset gedownload", state="complete")

ensure_dataset()

# Download 1x indien nodig
if not os.path.exists(FILE):
    if not DATA_URL:
        st.error("DATA_URL ontbreekt. Zet DATA_URL in Streamlit Secrets.")
        st.stop()
    with st.status("Dataset downloaden…", expanded=True) as s:
        st.write("Download URL (ingekort):", DATA_URL[:80] + "…")
        download_file(DATA_URL, FILE)
        s.update(label="Dataset gedownload", state="complete")


# In jouw data:
LON_COL = "GPS_x"   # longitude
LAT_COL = "GPS_y"   # latitude

# kolommen die geen "signalen" zijn (metadata)
EXCLUDE = {"Time", "Seconds", "Minutes", "Hours", "Year", "Month", "Day"}

# performance: max punten tekenen
MAX_POINTS = 8000


# =======================
# Helpers
# =======================
@st.cache_data
def read_schema():
    """Lees enkel schema (kolomnamen + types) zonder de hele dataset te laden."""
    pf = pq.ParquetFile(FILE)
    schema = pf.schema_arrow
    col_names = schema.names
    col_types = {name: schema.field(name).type for name in col_names}
    return col_names, col_types


def is_numeric_or_bool_arrow(pa_type) -> bool:
    """True voor int/uint/float/bool Arrow types."""
    s = str(pa_type).lower()
    return (
        "int" in s
        or "uint" in s
        or "float" in s
        or "double" in s
        or "bool" in s
    )


def choose_default(candidates, preferred_list):
    """Kies eerste die voorkomt in preferred_list, anders eerste candidate."""
    cand_set = set(candidates)
    for p in preferred_list:
        if p in cand_set:
            return p
    return candidates[0] if candidates else None


@st.cache_data
def load_columns(columns):
    """Laad enkel gevraagde kolommen, parse Timestamp, sorteer."""
    df = pd.read_parquet(FILE, columns=columns, engine="pyarrow")

    # Timestamp robuust parsen
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # timezone weg (Streamlit slider wil tz-naive)
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    except Exception:
        pass

    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return df


def downsample_ordered(df, max_points=MAX_POINTS):
    """Downsample met behoud van volgorde (mooie lijnen)."""
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx]


def make_map_figure(lon, lat):
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode="lines",
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
    )
    return fig


def make_line_figure(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
    )
    return fig


# =======================
# UI
# =======================
st.set_page_config(layout="wide")
st.title("Geo + 3 signalen (TotaleTimetable_date)")

# Schema lezen
col_names, col_types = read_schema()

# Basis checks
required = ["Timestamp", LON_COL, LAT_COL]
missing = [c for c in required if c not in col_names]
if missing:
    st.error(f"Ontbrekende kolommen in Parquet: {missing}")
    st.stop()

# Signaal-kandidaten bepalen op basis van Arrow schema (snel, zonder data te laden)
sig_candidates = []
for c in col_names:
    if c in required or c in EXCLUDE:
        continue
    if is_numeric_or_bool_arrow(col_types.get(c)):
        sig_candidates.append(c)

if not sig_candidates:
    st.error("Geen numerieke/logische signalen gevonden in de Parquet.")
    st.stop()

# Defaults: probeer iets zinnigs
sig1_default = choose_default(sig_candidates, ["GPS_speed", "TCO1_VehicleSpeed", "EEC1_Speed"])
sig2_default = choose_default(sig_candidates, ["GPS_course", "EEC2_AccPed1Position", "ET1_CoolantTemperature"])
sig3_default = choose_default(sig_candidates, ["GPS_z", "AAI_Temperature1", "AMB_AirTemperature"])

# Dropdowns
c1, c2, c3 = st.columns(3)
with c1:
    sig1 = st.selectbox("Signaal 1", sig_candidates, index=sig_candidates.index(sig1_default))
with c2:
    sig2 = st.selectbox("Signaal 2", sig_candidates, index=sig_candidates.index(sig2_default))
with c3:
    sig3 = st.selectbox("Signaal 3", sig_candidates, index=sig_candidates.index(sig3_default))

# Data laden: enkel nodig
df = load_columns(["Timestamp", LON_COL, LAT_COL, sig1, sig2, sig3])

if len(df) < 2:
    st.error("Te weinig rijen na het laden/parsen van Timestamp.")
    st.stop()

tmin_ts = df["Timestamp"].iloc[0]
tmax_ts = df["Timestamp"].iloc[-1]

# Streamlit slider wil python datetime (geen pandas Timestamp)
tmin = tmin_ts.to_pydatetime()
tmax = tmax_ts.to_pydatetime()

start_dt, end_dt = st.slider(
    "Tijdvenster (start/einde)",
    min_value=tmin,
    max_value=tmax,
    value=(tmin, tmax),
)

# terug naar pandas voor filtering
start = pd.Timestamp(start_dt)
end = pd.Timestamp(end_dt)

sub = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
if len(sub) < 2:
    st.warning("Te weinig data in dit tijdvenster.")
    st.stop()

# Downsample voor plots
subp = downsample_ordered(sub, MAX_POINTS)

# =======================
# Kaart
# =======================
st.subheader("GPS track (geselecteerd tijdvenster)")

sub_ok = subp.dropna(subset=[LON_COL, LAT_COL])
if len(sub_ok) < 2:
    st.info("Geen (voldoende) geldige GPS punten in deze selectie.")
else:
    fig_map = make_map_figure(sub_ok[LON_COL], sub_ok[LAT_COL])
    st.plotly_chart(fig_map, use_container_width=True)

# =======================
# 3 grafieken
# =======================
st.subheader("Signalen (geselecteerd tijdvenster)")

st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig1], sig1), use_container_width=True)
st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig2], sig2), use_container_width=True)

st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig3], sig3), use_container_width=True)


