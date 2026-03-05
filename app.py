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


def make_line_figure(x, y, title, selected_ts=None):
    """
    Lijn + klikbare (quasi onzichtbare) markers.
    + Verticale selectielijn (shape) over de volledige plothoogte.
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

    # DUIDELIJKE selectielijn
    if selected_ts is not None:
        ts = pd.to_datetime(selected_ts)
        fig.add_shape(
            type="line",
            x0=ts, x1=ts,
            y0=0, y1=1,
            xref="x", yref="paper",
            line={"width": 3},
            layer="above",
        )

    return fig


def safe_index(options, value, fallback):
    if value in options:
        return options.index(value)
    if fallback in options:
        return options.index(fallback)
    return 0


def nearest_values_at_multi(df_full, ts, signals):
    """
    Dichtstbijzijnde timestamp (geen interpolatie) + waarden voor alle gekozen signalen.
    """
    if ts is None or df_full.empty:
        return None

    cols = ["Timestamp"] + list(signals)
    d = df_full[cols].dropna(subset=["Timestamp"]).sort_values("Timestamp")
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

    out = {"Timestamp": row["Timestamp"].to_pydatetime()}
    for s in signals:
        out[s] = float(row[s]) if pd.notna(row[s]) else np.nan
    return out


def plot_and_capture_click(fig, key):
    """
    Klik op (bijna) eender waar op de lijn => selecteert een punt (door markers).
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
# UI
# =======================
st.set_page_config(layout="wide")
st.title("Geo + signalen (TotaleTimetable_date)")

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

# Defaults (SIG1 en SIG3 omgewisseld t.o.v. vroeger)
sig1_default = choose_default(sig_candidates, ["EEC1_Speed"])          # was GPS_speed
sig2_default = choose_default(sig_candidates, ["Verbruik_g_per_km"])
sig3_default = choose_default(sig_candidates, ["GPS_speed"])          # was EEC1_Speed

st.subheader("Signalen kiezen")

max_n = max(1, min(len(sig_candidates), 12))  # cap op 12 om UI netjes te houden
n_signals = st.slider("Aantal signalen in grafieken", 1, max_n, value=min(3, max_n), step=1)

# Dynamische selectboxes (zelfde gedrag als de vaste 3: typ-zoek, met session_state via key)
selected_signals = []
defaults = [sig1_default, sig2_default, sig3_default]

for i in range(n_signals):
    default_i = defaults[i] if i < len(defaults) else None

    # fallback: neem eerste niet-gebruikte kandidaat als default als default_i None of al gebruikt
    if default_i is None or default_i in selected_signals:
        for c in sig_candidates:
            if c not in selected_signals:
                default_i = c
                break

    key = f"sig{i+1}"  # sig1, sig2, sig3, ...
    label = f"Signaal {i+1} (typ om te zoeken)"
    value_for_index = st.session_state.get(key, default_i)
    idx = safe_index(sig_candidates, value_for_index, default_i)

    chosen = st.selectbox(label, sig_candidates, index=idx, key=key)
    selected_signals.append(chosen)

# Uniekheidscheck (zoals bij de vaste 3)
if len(set(selected_signals)) != len(selected_signals):
    st.warning("Kies elk signaal maar één keer (geen duplicaten).")
    st.stop()

# Data laden (GPS + alle gekozen signalen)
df = load_columns(["Timestamp", LON_COL, LAT_COL] + selected_signals)

if len(df) < 2:
    st.error("Te weinig rijen na laden/parsen van Timestamp.")
    st.stop()

tmin_ts = df["Timestamp"].iloc[0]
tmax_ts = df["Timestamp"].iloc[-1]
tmin = tmin_ts.to_pydatetime()
tmax = tmax_ts.to_pydatetime()

# Fijnere slider stap
total_seconds = max(1, int((tmax - tmin).total_seconds()))
step_seconds = max(1, total_seconds // 20000)  # ~20.000 stapjes
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
# Export geselecteerd tijdvenster (CSV) - neemt alle gekozen signalen mee
# =======================
export_cols = ["Timestamp", LON_COL, LAT_COL] + selected_signals
export_df = sub[export_cols].copy()

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
# Grafieken + selectielijn + waarden
# =======================
st.subheader("Signalen (geselecteerd tijdvenster)")

if "clicked_ts" not in st.session_state:
    st.session_state["clicked_ts"] = None

selected_ts = st.session_state["clicked_ts"]

# Maak & toon grafieken (verticaal onder elkaar)
clicked_any = None
for i, sig in enumerate(selected_signals, start=1):
    fig = make_line_figure(subp["Timestamp"], subp[sig], sig, selected_ts=selected_ts)
    clicked = plot_and_capture_click(fig, key=f"plot_sig{i}")
    clicked_any = clicked_any or clicked

if clicked_any is not None:
    st.session_state["clicked_ts"] = clicked_any
    selected_ts = clicked_any

# Waarden tonen voor geselecteerde timestamp
vals = nearest_values_at_multi(df, selected_ts, selected_signals)

with st.container(border=True):
    st.markdown("### Meetpunt (klik in een grafiek)")
    if vals is None:
        st.write("Klik ergens op de lijn in een grafiek om het tijdstip te selecteren.")
        st.caption("De verticale selectielijn verschijnt op alle grafieken.")
    else:
        st.write(f"**Timestamp:** {vals['Timestamp']}")

        # metrics in rijen van 3
        per_row = 3
        for r in range(0, len(selected_signals), per_row):
            row_sigs = selected_signals[r:r + per_row]
            cols = st.columns(len(row_sigs))
            for c, s in zip(cols, row_sigs):
                with c:
                    st.metric(s, f"{vals[s]:.6f}")
