import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq
import gdown

# =======================
# Config
# =======================
FILE = "TotaleTimetable_date.parquet"

LON_COL = "GPS_x"   # longitude (graden)
LAT_COL = "GPS_y"   # latitude  (graden)

EXCLUDE = {"Time", "Seconds", "Minutes", "Hours", "Year", "Month", "Day"}
MAX_POINTS = 8000

# Secret in Streamlit Cloud
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "").strip()


# =======================
# Download dataset (Google Drive)
# =======================
def ensure_dataset():
    if os.path.exists(FILE):
        return

    if not GDRIVE_FILE_ID:
        st.error("Secret GDRIVE_FILE_ID ontbreekt (App settings → Secrets).")
        st.stop()

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download"
    with st.status("Dataset downloaden van Google Drive…", expanded=True) as s:
        st.write("Drive file id:", GDRIVE_FILE_ID)
        try:
            gdown.download(url, FILE, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Download faalde: {type(e).__name__}: {e}")
            st.stop()
        s.update(label="Dataset gedownload", state="complete")


# =======================
# Helpers
# =======================
@st.cache_data
def read_schema():
    pf = pq.ParquetFile(FILE)
    schema = pf.schema_arrow
    col_names = schema.names
    col_types = {name: schema.field(name).type for name in col_names}
    return col_names, col_types


def is_numeric_or_bool_arrow(pa_type) -> bool:
    s = str(pa_type).lower()
    return ("int" in s) or ("uint" in s) or ("float" in s) or ("double" in s) or ("bool" in s)


def choose_default(candidates, preferred_list):
    cand_set = set(candidates)
    for p in preferred_list:
        if p in cand_set:
            return p
    return candidates[0] if candidates else None


@st.cache_data
def load_columns(columns):
    df = pd.read_parquet(FILE, columns=columns, engine="pyarrow")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # tz-naive (Streamlit slider wil dit)
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    except Exception:
        pass

    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return df


def downsample_ordered(df, max_points=MAX_POINTS):
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx]


def make_map_figure(sub_ok, start_dt, end_dt):
    # Maak een duidelijke kaart: lijn + start/eind marker + fitbounds
    lon = sub_ok[LON_COL].to_numpy()
    lat = sub_ok[LAT_COL].to_numpy()

    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode="lines",
            name="track"
        )
    )

    # start marker
    fig.add_trace(
        go.Scattermapbox(
            lon=[lon[0]],
            lat=[lat[0]],
            mode="markers",
            marker={"size": 10},
            name="start"
        )
    )

    # end marker
    fig.add_trace(
        go.Scattermapbox(
            lon=[lon[-1]],
            lat=[lat[-1]],
            mode="markers",
            marker={"size": 10},
            name="einde"
        )
    )

    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "fitbounds": "locations",   # <-- zorgt dat de selectie altijd in beeld komt
        },
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        title=f"GPS track: {start_dt} → {end_dt}",
        showlegend=False
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

ensure_dataset()

col_names, col_types = read_schema()

required = ["Timestamp", LON_COL, LAT_COL]
missing = [c for c in required if c not in col_names]
if missing:
    st.error(f"Ontbrekende kolommen in Parquet: {missing}")
    st.stop()

# signaal-kandidaten
sig_candidates = []
for c in col_names:
    if c in required or c in EXCLUDE:
        continue
    if is_numeric_or_bool_arrow(col_types.get(c)):
        sig_candidates.append(c)

if not sig_candidates:
    st.error("Geen numerieke/logische signalen gevonden in de Parquet.")
    st.stop()

sig1_default = choose_default(sig_candidates, ["GPS_speed", "TCO1_VehicleSpeed", "EEC1_Speed"])
sig2_default = choose_default(sig_candidates, ["GPS_course", "EEC2_AccPed1Position", "ET1_CoolantTemperature"])
sig3_default = choose_default(sig_candidates, ["GPS_z", "AAI_Temperature1", "AMB_AirTemperature"])

c1, c2, c3 = st.columns(3)
with c1:
    sig1 = st.selectbox("Signaal 1", sig_candidates, index=sig_candidates.index(sig1_default))
with c2:
    sig2 = st.selectbox("Signaal 2", sig_candidates, index=sig_candidates.index(sig2_default))
with c3:
    sig3 = st.selectbox("Signaal 3", sig_candidates, index=sig_candidates.index(sig3_default))

df = load_columns(["Timestamp", LON_COL, LAT_COL, sig1, sig2, sig3])

if len(df) < 2:
    st.error("Te weinig rijen na laden/parsen van Timestamp.")
    st.stop()

tmin_ts = df["Timestamp"].iloc[0]
tmax_ts = df["Timestamp"].iloc[-1]

# Streamlit slider wil python datetime
tmin = tmin_ts.to_pydatetime()
tmax = tmax_ts.to_pydatetime()

start_dt, end_dt = st.slider(
    "Tijdvenster (start/einde)",
    min_value=tmin,
    max_value=tmax,
    value=(tmin, tmax),
)

start = pd.Timestamp(start_dt)
end = pd.Timestamp(end_dt)

sub = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
if len(sub) < 2:
    st.warning("Te weinig data in dit tijdvenster.")
    st.stop()

subp = downsample_ordered(sub, MAX_POINTS)

# =======================
# Kaart
# =======================
st.subheader("GPS track (geselecteerd tijdvenster)")

# Forceer GPS naar numeriek (dit is vaak de echte oorzaak dat er niets verschijnt)
subp[LON_COL] = pd.to_numeric(subp[LON_COL], errors="coerce")
subp[LAT_COL] = pd.to_numeric(subp[LAT_COL], errors="coerce")

sub_ok = subp.dropna(subset=[LON_COL, LAT_COL])
sub_ok = sub_ok[np.isfinite(sub_ok[LON_COL]) & np.isfinite(sub_ok[LAT_COL])]

if len(sub_ok) < 2:
    st.info("Geen (voldoende) geldige GPS punten in deze selectie.")
else:
    # sanity check: lon/lat moeten in graden liggen
    lon_ok = sub_ok[LON_COL].between(-180, 180).all()
    lat_ok = sub_ok[LAT_COL].between(-90, 90).all()
    if not (lon_ok and lat_ok):
        st.warning(
            "GPS_x/GPS_y lijken geen lon/lat in graden te zijn (waarden buiten [-180,180]/[-90,90]). "
            "Dan werkt Scattermapbox niet. Check of GPS_x/GPS_y misschien UTM/meters zijn, of omgewisseld."
        )

    fig_map = make_map_figure(sub_ok, start_dt, end_dt)

    # key mee laten veranderen met slider => Streamlit forceert her-render van de kaart
    map_key = f"map-{start_dt.isoformat()}-{end_dt.isoformat()}"
    st.plotly_chart(fig_map, use_container_width=True, key=map_key)

# =======================
# 3 grafieken
# =======================
st.subheader("Signalen (geselecteerd tijdvenster)")
st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig1], sig1), use_container_width=True)
st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig2], sig2), use_container_width=True)
st.plotly_chart(make_line_figure(subp["Timestamp"], subp[sig3], sig3), use_container_width=True)
