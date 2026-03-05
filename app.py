import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import gdown

# =======================
# Config
# =======================
FILE = "TotaleTimetable_date.parquet"

LAT_COL = "GPS_x"   # longitude (graden)
LON_COL = "GPS_y"   # latitude  (graden)

EXCLUDE = {"Time", "Seconds", "Minutes", "Hours", "Year", "Month", "Day"}

MAX_POINTS = 8000
TIME_STEP = timedelta(minutes=1)     # slider stap
BIG_WINDOW = timedelta(hours=1)      # > 1 uur => warning + confirm export

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


def downsample_ordered(df, max_points=MAX_POINTS):
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx]


def safe_index(options, value, fallback):
    if value in options:
        return options.index(value)
    if fallback in options:
        return options.index(fallback)
    return 0


def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


@st.cache_data(show_spinner=False)
def get_time_bounds_from_metadata() -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Snel min/max Timestamp uit Parquet row-group stats.
    Fallback: Timestamp-kolom lezen.
    """
    pf = pq.ParquetFile(FILE)

    try:
        ts_col_idx = pf.schema_arrow.get_field_index("Timestamp")
    except Exception:
        ts_col_idx = None

    tmin = None
    tmax = None

    if ts_col_idx is not None and ts_col_idx >= 0 and pf.metadata is not None:
        md = pf.metadata
        for rg in range(md.num_row_groups):
            col = md.row_group(rg).column(ts_col_idx)
            stats = col.statistics
            if stats is None:
                continue
            try:
                rg_min = pd.to_datetime(stats.min)
                rg_max = pd.to_datetime(stats.max)
            except Exception:
                continue
            if pd.notna(rg_min):
                tmin = rg_min if (tmin is None or rg_min < tmin) else tmin
            if pd.notna(rg_max):
                tmax = rg_max if (tmax is None or rg_max > tmax) else tmax

    if tmin is None or tmax is None:
        df_ts = pd.read_parquet(FILE, columns=["Timestamp"], engine="pyarrow")
        df_ts["Timestamp"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce")
        try:
            df_ts["Timestamp"] = df_ts["Timestamp"].dt.tz_localize(None)
        except Exception:
            pass
        df_ts = df_ts.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        tmin = df_ts["Timestamp"].iloc[0]
        tmax = df_ts["Timestamp"].iloc[-1]

    tmin = pd.Timestamp(tmin).tz_localize(None) if getattr(pd.Timestamp(tmin), "tzinfo", None) else pd.Timestamp(tmin)
    tmax = pd.Timestamp(tmax).tz_localize(None) if getattr(pd.Timestamp(tmax), "tzinfo", None) else pd.Timestamp(tmax)
    return tmin, tmax


@st.cache_data(show_spinner=False)
def load_filtered(columns: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Leest enkel gevraagde kolommen en enkel rijen in tijdvenster (filter pushdown).
    """
    dataset = ds.dataset(FILE, format="parquet")
    filt = (ds.field("Timestamp") >= ds.scalar(start.to_pydatetime())) & (ds.field("Timestamp") <= ds.scalar(end.to_pydatetime()))
    table = dataset.to_table(columns=columns, filter=filt)

    df = table.to_pandas()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return df


@st.cache_data(show_spinner=False)
def load_preview_sample_full_range(signal: str, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """
    Preview over VOLLEDIG tijdsbereik (downsampled) zonder alles in RAM te laden.
    Reservoir sampling op Timestamp + 1 signaal, daarna sorteren op tijd.
    """
    dataset = ds.dataset(FILE, format="parquet")
    cols = ["Timestamp", signal]

    rng = np.random.default_rng(12345)
    reservoir_ts = []
    reservoir_val = []
    seen = 0

    # ✅ compatibel: scanner() i.p.v. scan()
    scanner = dataset.scanner(columns=cols, batch_size=250_000)
    for batch in scanner.to_batches():
        ts_arr = batch.column(0).to_pylist()
        val_arr = batch.column(1).to_pylist()

        for ts, v in zip(ts_arr, val_arr):
            if ts is None:
                continue
            seen += 1
            if len(reservoir_ts) < max_points:
                reservoir_ts.append(ts)
                reservoir_val.append(v)
            else:
                j = int(rng.integers(0, seen))
                if j < max_points:
                    reservoir_ts[j] = ts
                    reservoir_val[j] = v

    if not reservoir_ts:
        return pd.DataFrame(columns=["Timestamp", signal])

    df = pd.DataFrame({"Timestamp": pd.to_datetime(reservoir_ts, errors="coerce"), signal: reservoir_val})
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    return df


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
    fig.add_trace(go.Scattermapbox(lon=[lon[0]], lat=[lat[0]], mode="markers", marker={"size": 10}, name="start"))
    fig.add_trace(go.Scattermapbox(lon=[lon[-1]], lat=[lat[-1]], mode="markers", marker={"size": 10}, name="einde"))

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker={"size": 8, "opacity": 0.01},
        line={"width": 2},
        name=title
    ))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
        hovermode="x",
        showlegend=False,
        clickmode="event+select",
    )

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


def make_preview_figure_with_window(preview_df: pd.DataFrame, signal: str, start_dt: datetime, end_dt: datetime):
    """
    Previewfiguur over volledig bereik + visuele selectie van tijdslot (shaded region + 2 lijnen).
    """
    fig = make_line_figure(preview_df["Timestamp"], preview_df[signal], f"Preview: {signal}")

    # Shaded selectie over volledige y-range
    fig.add_vrect(
        x0=pd.Timestamp(start_dt),
        x1=pd.Timestamp(end_dt),
        fillcolor="rgba(0,0,0,0.12)",  # neutraal, licht
        line_width=0,
        layer="below",
    )
    # Duidelijke randen
    fig.add_vline(x=pd.Timestamp(start_dt), line_width=2)
    fig.add_vline(x=pd.Timestamp(end_dt), line_width=2)

    fig.update_layout(
        title=f"Preview: {signal}  (selectie: {start_dt} → {end_dt})"
    )
    return fig


def nearest_values_at_multi(df_full, ts, signals):
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
    ev = st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode=["points"],
    )

    sel = ev.get("selection") if isinstance(ev, dict) else getattr(ev, "selection", None)
    if not sel:
        return None

    points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not points:
        return None

    p0 = points[0]
    return p0.get("x") if isinstance(p0, dict) else getattr(p0, "x", None)


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

sig_candidates = []
for c in col_names:
    if c in required or c in EXCLUDE:
        continue
    if is_numeric_or_bool_arrow(col_types.get(c)):
        sig_candidates.append(c)

if not sig_candidates:
    st.error("Geen numerieke/logische signalen gevonden in de Parquet.")
    st.stop()

# Defaults (SIG1 en SIG3 omgewisseld)
sig1_default = choose_default(sig_candidates, ["EEC1_Speed"])
sig2_default = choose_default(sig_candidates, ["Verbruik_g_per_km"])
sig3_default = choose_default(sig_candidates, ["GPS_speed"])

# =======================
# Tijdvenster kiezen (voor selectie in preview + export + laden)
# =======================
st.subheader("Tijdvenster kiezen")

tmin_ts, tmax_ts = get_time_bounds_from_metadata()
tmin = floor_to_minute(tmin_ts.to_pydatetime())
tmax = floor_to_minute(tmax_ts.to_pydatetime())
step = TIME_STEP


def clamp_dt(dt: datetime) -> datetime:
    if dt < tmin:
        return tmin
    if dt > tmax:
        return tmax
    return dt


def ensure_valid_range(s: datetime, e: datetime) -> tuple[datetime, datetime]:
    s = floor_to_minute(clamp_dt(s))
    e = floor_to_minute(clamp_dt(e))
    if s >= e:
        e = floor_to_minute(clamp_dt(s + step))
    return s, e


def update_from_slider():
    s, e = st.session_state["time_slider"]
    s, e = ensure_valid_range(s, e)

    st.session_state["start_dt"] = s
    st.session_state["end_dt"] = e
    st.session_state["start_date"] = s.date()
    st.session_state["start_time"] = s.time().replace(second=0, microsecond=0)
    st.session_state["end_date"] = e.date()
    st.session_state["end_time"] = e.time().replace(second=0, microsecond=0)


def update_from_inputs():
    s = datetime.combine(st.session_state["start_date"], st.session_state["start_time"])
    e = datetime.combine(st.session_state["end_date"], st.session_state["end_time"])
    s, e = ensure_valid_range(s, e)

    st.session_state["start_dt"] = s
    st.session_state["end_dt"] = e
    st.session_state["time_slider"] = (s, e)


# init state: start bij begin dataset, eind = begin + 1 uur (licht)
if "start_dt" not in st.session_state or "end_dt" not in st.session_state:
    default_start = tmin
    default_end = min(tmax, default_start + BIG_WINDOW)

    st.session_state["start_dt"] = default_start
    st.session_state["end_dt"] = default_end
    st.session_state["start_date"] = default_start.date()
    st.session_state["start_time"] = default_start.time().replace(second=0, microsecond=0)
    st.session_state["end_date"] = default_end.date()
    st.session_state["end_time"] = default_end.time().replace(second=0, microsecond=0)
    st.session_state["time_slider"] = (default_start, default_end)

cA, cB = st.columns(2)
with cA:
    st.date_input("Start datum", key="start_date", on_change=update_from_inputs)
    st.time_input("Start tijd", key="start_time", step=60, on_change=update_from_inputs)  # ✅ 1 minuut
with cB:
    st.date_input("Eind datum", key="end_date", on_change=update_from_inputs)
    st.time_input("Eind tijd", key="end_time", step=60, on_change=update_from_inputs)    # ✅ 1 minuut

st.slider(
    "Tijdvenster (start/einde)",
    min_value=tmin,
    max_value=tmax,
    value=(st.session_state["start_dt"], st.session_state["end_dt"]),
    step=step,
    key="time_slider",
    on_change=update_from_slider,
)

start_dt = st.session_state["start_dt"]
end_dt = st.session_state["end_dt"]
start = pd.Timestamp(start_dt)
end = pd.Timestamp(end_dt)
duration = end - start

# =======================
# Preview: volledig signaal + toon selectie (slider-range) in de grafiek
# =======================
st.divider()
st.subheader("Preview (volledig signaal + selectie zichtbaar)")

preview_signal = st.selectbox(
    "Preview signaal",
    sig_candidates,
    index=safe_index(sig_candidates, st.session_state.get("preview_signal", sig1_default), sig1_default),
    key="preview_signal",
)

preview_df = load_preview_sample_full_range(preview_signal, max_points=MAX_POINTS)
if len(preview_df) < 2:
    st.warning("Te weinig data om preview te tonen.")
else:
    st.caption("De grijze zone in de preview toont het gekozen tijdslot (via sliders/tijdvakken).")
    st.plotly_chart(
        make_preview_figure_with_window(preview_df, preview_signal, start_dt, end_dt),
        width="stretch",
    )

# =======================
# Signalen kiezen voor grafieken & download
# =======================
st.divider()
st.subheader("Signalen kiezen voor grafieken & download")

max_n = max(1, min(len(sig_candidates), 12))
n_signals = st.slider("Aantal signalen", 1, max_n, value=min(3, max_n), step=1)

selected_signals = []
defaults = [sig1_default, sig2_default, sig3_default]

for i in range(n_signals):
    default_i = defaults[i] if i < len(defaults) else None
    if default_i is None or default_i in selected_signals:
        for c in sig_candidates:
            if c not in selected_signals:
                default_i = c
                break

    key = f"sig{i+1}"
    label = f"Signaal {i+1} (typ om te zoeken)"
    value_for_index = st.session_state.get(key, default_i)
    idx = safe_index(sig_candidates, value_for_index, default_i)

    chosen = st.selectbox(label, sig_candidates, index=idx, key=key)
    selected_signals.append(chosen)

if len(set(selected_signals)) != len(selected_signals):
    st.warning("Kies elk signaal maar één keer (geen duplicaten).")
    st.stop()

# ✅ Nooit volledige dataset: enkel Timestamp + GPS + gekozen signalen, en enkel in venster
cols_needed = ["Timestamp", LON_COL, LAT_COL] + selected_signals

# =======================
# Download (CSV) - enkel geselecteerde signalen
# =======================
st.subheader("Download (CSV)")

start_str = pd.Timestamp(start_dt).strftime("%Y%m%d_%H%M%S")
end_str = pd.Timestamp(end_dt).strftime("%Y%m%d_%H%M%S")

if duration > BIG_WINDOW:
    st.warning(
        "Je geselecteerde tijdslot is groter dan 1 uur. "
        "Een grotere export kan traag zijn of problemen geven in de app.\n\n"
        "Standaard kan je hieronder eerst **enkel het eerste uur** downloaden. "
        "Wil je toch het volledige (grotere) tijdslot, vink dan bevestiging aan."
    )

    first_hour_end_dt = min(end_dt, start_dt + BIG_WINDOW)
    first_hour_end = pd.Timestamp(first_hour_end_dt)

    export_first = load_filtered(cols_needed, start, first_hour_end)
    csv_name_1h = f"selectie_EERSTE_UUR_{start_str}_tot_{pd.Timestamp(first_hour_end_dt).strftime('%Y%m%d_%H%M%S')}.csv"
    csv_bytes_1h = export_first.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download eerste uur (veilig)",
        data=csv_bytes_1h,
        file_name=csv_name_1h,
        mime="text/csv",
    )

    confirm_key = "confirm_big_export"
    st.checkbox(
        "Ik begrijp dat een export > 1 uur problemen kan geven en wil toch het volledige tijdslot downloaden.",
        key=confirm_key,
    )

    if st.session_state.get(confirm_key, False):
        export_full = load_filtered(cols_needed, start, end)
        csv_name_full = f"selectie_VOLLEDIG_{start_str}_tot_{end_str}.csv"
        csv_bytes_full = export_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download volledig tijdslot (kan zwaar zijn)",
            data=csv_bytes_full,
            file_name=csv_name_full,
            mime="text/csv",
        )
    else:
        st.info("Vink de bevestiging aan om het volledige tijdslot (> 1 uur) te kunnen downloaden.")
else:
    export_df = load_filtered(cols_needed, start, end)
    csv_name = f"selectie_{start_str}_tot_{end_str}.csv"
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download geselecteerd tijdvenster (CSV)",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
    )

# =======================
# Visualisatie pas na knop
# =======================
st.subheader("Visualisatie")

if "load_confirmed" not in st.session_state:
    st.session_state["load_confirmed"] = False

if duration > BIG_WINDOW:
    st.warning("Je geselecteerde tijdslot is groter dan 1 uur. Dit kan traag zijn of problemen geven in de app.")

if st.button("✅ Laad nu data en toon kaart + grafieken", type="primary"):
    st.session_state["load_confirmed"] = True

if not st.session_state["load_confirmed"]:
    st.info("Klik op **Laad nu data** om kaart en grafieken te tonen.")
    st.stop()

df = load_filtered(cols_needed, start, end)
if len(df) < 2:
    st.warning("Te weinig data in dit tijdvenster.")
    st.stop()

subp = downsample_ordered(df, MAX_POINTS).copy()

# Kaart
st.subheader("GPS track (geselecteerd tijdvenster)")
subp.loc[:, LON_COL] = pd.to_numeric(subp[LON_COL], errors="coerce")
subp.loc[:, LAT_COL] = pd.to_numeric(subp[LAT_COL], errors="coerce")

sub_ok = subp.dropna(subset=[LON_COL, LAT_COL])
sub_ok = sub_ok[np.isfinite(sub_ok[LON_COL]) & np.isfinite(sub_ok[LAT_COL])]

if len(sub_ok) < 2:
    st.info("Geen (voldoende) geldige GPS punten in deze selectie.")
else:
    st.plotly_chart(make_map_figure(sub_ok, start_dt, end_dt), width="stretch")

# Grafieken + selectielijn
st.subheader("Signalen (geselecteerd tijdvenster)")
if "clicked_ts" not in st.session_state:
    st.session_state["clicked_ts"] = None

selected_ts = st.session_state["clicked_ts"]

clicked_any = None
for i, sig in enumerate(selected_signals, start=1):
    fig = make_line_figure(subp["Timestamp"], subp[sig], sig, selected_ts=selected_ts)
    clicked = plot_and_capture_click(fig, key=f"plot_sig{i}")
    if clicked_any is None and clicked is not None:
        clicked_any = clicked

if clicked_any is not None:
    st.session_state["clicked_ts"] = clicked_any
    selected_ts = clicked_any

vals = nearest_values_at_multi(df, selected_ts, selected_signals)

with st.container(border=True):
    st.markdown("### Meetpunt (klik in een grafiek)")
    if vals is None:
        st.write("Klik ergens op de lijn in een grafiek om het tijdstip te selecteren.")
        st.caption("De verticale selectielijn verschijnt op alle grafieken.")
    else:
        st.write(f"**Timestamp:** {vals['Timestamp']}")
        per_row = 3
        for r in range(0, len(selected_signals), per_row):
            row_sigs = selected_signals[r:r + per_row]
            cols = st.columns(len(row_sigs))
            for c, s in zip(cols, row_sigs):
                with c:
                    st.metric(s, f"{vals[s]:.6f}")
