import os
from datetime import timedelta

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

LAT_COL = "GPS_x"   # longitude (graden)
LON_COL = "GPS_y"   # latitude  (graden)

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
    lon = sub_ok[LON_COL].to_numpy(dtype=float)
    lat = sub_ok[LAT_COL].to_numpy(dtype=float)

    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

    center_lon = float((lon_min + lon_max) / 2.0)
    center_lat = float((lat_min + lat_max) / 2.0)

    lon_span = float(max(lon_max - lon_min, 1e-6))
    lat_span = float(max(lat_max - lat_min, 1e-6))

    zoom_lon = np.log2(360.0 / lon_span)
    zoom_lat = np.log2(180.0 / lat_span)
    zoom = float(np.clip(min(zoom_lon, zoom_lat) - 1.0, 1.0, 18.0))

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(lon=lon, lat=lat, mode="lines", name="track"))
    fig.add_trace(go.Scattermapbox(lon=[lon[0]], lat=[lat[0]], mode="markers",
                                   marker={"size": 10}, name="start"))
    fig.add_trace(go.Scattermapbox(lon=[lon[-1]], lat=[lat[-1]], mode="markers",
                                   marker={"size": 10}, name="einde"))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom,
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        title=f"GPS track: {start_dt} → {end_dt}",
        showlegend=False,
    )
    return fig


def make_line_figure(x, y, title):
    """
    Lijnplot + onzichtbare markers zodat select-events altijd klikbare punten hebben.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        marker={"size": 8, "opacity": 0.01},  # quasi onzichtbaar maar klikbaar
        line={"width": 2},
        name=title
    ))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
        hovermode="x",
        showlegend=False,
        clickmode="event+select",
    )
    return fig


def safe_index(options, value, fallback):
    if value in options:
        return options.index(value)
    if fallback in options:
        return options.index(fallback)
    return 0


def add_vline(fig, x):
    if x is None:
        return fig
    fig.add_vline(x=pd.to_datetime(x), line_width=2)
    return fig


def nearest_values_at(df_full, ts, sig1, sig2, sig3):
    """
    Dichtstbijzijnde timestamp (geen interpolatie).
    """
    if ts is None or df_full.empty:
        return None

    d = df_full[["Timestamp", sig1, sig2, sig3]].dropna(subset=["Timestamp"]).sort_values("Timestamp")
    t = pd.Timestamp(ts)

    idx = d["Timestamp"].searchsorted(t)

    if idx <= 0:
        row = d.iloc[0]
    elif idx >= len(d):
        row = d.iloc[-1]
    else:
        before = d.iloc[idx - 1]
        after = d.iloc[idx]
        row = before if (t - before["Timestamp"]) <= (after["Timestamp"] - t) else after

    return {
        "Timestamp": row["Timestamp"].to_pydatetime(),
        sig1: float(row[sig1]) if pd.notna(row[sig1]) else np.nan,
        sig2: float(row[sig2]) if pd.notna(row[sig2]) else np.nan,
        sig3: float(row[sig3]) if pd.notna(row[sig3]) else np.nan,
    }


def plot_and_capture_click(fig, key):
    """
    Laat je "eender waar" klikken:
    - We lezen de clickData uit plotly via select-event.
    - Door onzichtbare markers is een klik op de lijn (bijna altijd) een punt-selectie.
    """
    ev = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode=["points"],
    )

    sel = None
    if isinstance(ev, dict):
        sel = ev.get("selection")
    else:
        sel = getattr(ev, "selection", None)

    if not sel:
        return None

    points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not points:
        return None

    p0 = points[0]
    if isinstance(p0, dict):
        return p0.get("x")
    return getattr(p0, "x", None)


# =======================
# Export helpers
# =======================
def make_csv_bytes(df_export: pd.DataFrame) -> bytes:
    # UTF-8 with BOM => Excel vriendelijk
    return df_export.to_csv(index=False).encode("utf-8-sig")


def export_filename(start_dt, end_dt) -> str:
    s = pd.Timestamp(start_dt).strftime("%Y%m%d_%H%M%S")
    e = pd.Timestamp(end_dt).strftime("%Y%m%d_%H%M%S")
    return f"tijdsvenster_{s}_tot_{e}.csv"


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

# -----------------------
# Nieuwe gewenste defaults
# -----------------------
sig1_default = choose_default(sig_candidates, ["GPS_speed"])
sig2_default = choose_default(sig_candidates, ["Verbruik_g_per_km"])
sig3_default = choose_default(sig_candidates, ["EEC1_Speed"])

# Keuze onthouden tussen reruns
if "sig1" not in st.session_state:
    st.session_state["sig1"] = sig1_default
if "sig2" not in st.session_state:
    st.session_state["sig2"] = sig2_default
if "sig3" not in st.session_state:
    st.session_state["sig3"] = sig3_default

c1, c2, c3 = st.columns(3)
with c1:
    sig1 = st.selectbox(
        "Signaal 1 (typ om te zoeken)",
        sig_candidates,
        index=safe_index(sig_candidates, st.session_state["sig1"], sig1_default),
        key="sig1",
    )
with c2:
    sig2 = st.selectbox(
        "Signaal 2 (typ om te zoeken)",
        sig_candidates,
        index=safe_index(sig_candidates, st.session_state["sig2"], sig2_default),
        key="sig2",
    )
with c3:
    sig3 = st.selectbox(
        "Signaal 3 (typ om te zoeken)",
        sig_candidates,
        index=safe_index(sig_candidates, st.session_state["sig3"], sig3_default),
        key="sig3",
    )

# Data laden
df = load_columns(["Timestamp", LON_COL, LAT_COL, sig1, sig2, sig3])

if len(df) < 2:
    st.error("Te weinig rijen na laden/parsen van Timestamp.")
    st.stop()

tmin_ts = df["Timestamp"].iloc[0]
tmax_ts = df["Timestamp"].iloc[-1]
tmin = tmin_ts.to_pydatetime()
tmax = tmax_ts.to_pydatetime()

# Fijnere slider stap
total_seconds = max(1, int((tmax - tmin).total_seconds()))
step_seconds = max(1, total_seconds // 20000)  # nog fijner (~20.000 stapjes)
step_seconds = min(step_seconds, 10)           # nooit grover dan 10s
step = timedelta(seconds=step_seconds)

start_dt, end_dt = st.slider(
    "Tijdvenster (start/einde)",
    min_value=tmin,
    max_value=tmax,
    value=(tmin, tmax),
    step=step,
)

start = pd.Timestamp(start_dt)
end = pd.Timestamp(end_dt)

sub = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
if len(sub) < 2:
    st.warning("Te weinig data in dit tijdvenster.")
    st.stop()

# =======================
# Export (toegevoegd, zonder rest te breken)
# =======================
st.subheader("Export geselecteerd tijdvenster")

with st.container(border=True):
    export_cols = ["Timestamp", LON_COL, LAT_COL, sig1, sig2, sig3]
    export_df = sub[export_cols].copy()

    # Optioneel: Timestamp als string (Excel-vriendelijk); laat rest ongemoeid
    export_df["Timestamp"] = pd.to_datetime(export_df["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    csv_bytes = make_csv_bytes(export_df)
    fname = export_filename(start_dt, end_dt)

    st.download_button(
        label="⬇️ Export naar CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Na klikken opent je browser het download-/opslaan-venster (daar kies je locatie).")

# Downsample voor plots/kaart (zoals originele app)
subp = downsample_ordered(sub, MAX_POINTS)

# =======================
# Kaart
# =======================
st.subheader("GPS track (geselecteerd tijdvenster)")

subp[LON_COL] = pd.to_numeric(subp[LON_COL], errors="coerce")
subp[LAT_COL] = pd.to_numeric(subp[LAT_COL], errors="coerce")

sub_ok = subp.dropna(subset=[LON_COL, LAT_COL])
sub_ok = sub_ok[np.isfinite(sub_ok[LON_COL]) & np.isfinite(sub_ok[LAT_COL])]

if len(sub_ok) < 2:
    st.info("Geen (voldoende) geldige GPS punten in deze selectie.")
else:
    fig_map = make_map_figure(sub_ok, start_dt, end_dt)
    st.plotly_chart(fig_map, use_container_width=True)

# =======================
# 3 grafieken + meetlijn + waarden
# =======================
st.subheader("Signalen (geselecteerd tijdvenster)")

if "clicked_ts" not in st.session_state:
    st.session_state["clicked_ts"] = None

# figuren
fig1 = make_line_figure(subp["Timestamp"], subp[sig1], sig1)
fig2 = make_line_figure(subp["Timestamp"], subp[sig2], sig2)
fig3 = make_line_figure(subp["Timestamp"], subp[sig3], sig3)

# meetlijn op huidige clicked_ts
fig1 = add_vline(fig1, st.session_state["clicked_ts"])
fig2 = add_vline(fig2, st.session_state["clicked_ts"])
fig3 = add_vline(fig3, st.session_state["clicked_ts"])

# Klik op eender welke grafiek
clicked1 = plot_and_capture_click(fig1, key="plot_sig1")
clicked2 = plot_and_capture_click(fig2, key="plot_sig2")
clicked3 = plot_and_capture_click(fig3, key="plot_sig3")

new_click = clicked1 or clicked2 or clicked3
if new_click is not None:
    st.session_state["clicked_ts"] = new_click

# waardenkader (decimale cijfers)
vals = nearest_values_at(df, st.session_state["clicked_ts"], sig1, sig2, sig3)

with st.container(border=True):
    st.markdown("### Meetpunt (klik in een grafiek)")
    if vals is None:
        st.write("Klik ergens op de lijn in een grafiek om het tijdstip te selecteren.")
        st.caption("Door onzichtbare markers is een klik op de lijn selecteerbaar.")
    else:
        st.write(f"**Timestamp:** {vals['Timestamp']}")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric(sig1, f"{vals[sig1]:.6f}")
        with cc2:
            st.metric(sig2, f"{vals[sig2]:.6f}")
        with cc3:
            st.metric(sig3, f"{vals[sig3]:.6f}")
