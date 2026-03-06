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
TIME_STEP = timedelta(minutes=1)
BIG_WINDOW = timedelta(hours=1)

GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "").strip()


# =======================
# Download dataset (Google Drive) + fallback upload
# =======================
def ensure_dataset():
    """
    Zorgt dat FILE lokaal bestaat.
    - Als het al bestaat: ok
    - Anders: probeert via gdown te downloaden
    - Als Drive quota gdown blokkeert: toont upload fallback
    """
    if os.path.exists(FILE):
        return

    st.info("Dataset is nog niet lokaal aanwezig.")

    # 1) Probeer gdown indien file id aanwezig
    if GDRIVE_FILE_ID:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download"
        try:
            with st.status("Dataset downloaden van Google Drive…", expanded=True) as s:
                st.write("Drive file id:", GDRIVE_FILE_ID)
                gdown.download(url, FILE, quiet=False, fuzzy=True)
                s.update(label="Dataset gedownload", state="complete")
            return
        except Exception as e:
            # Drive quota / permissions / etc.
            st.error(f"Download faalde via gdown: {type(e).__name__}: {e}")

            # Toon browserlink
            st.write("Je kan mogelijk nog wél downloaden via de browser:")
            st.write(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}")

            st.warning(
                "Dit is typisch een Google Drive quota/traffic limit ('Too many users have viewed or downloaded'). "
                "Oplossingen: zet de file op een andere host (GCS/Azure/S3) of upload hem hier handmatig."
            )

    else:
        st.warning("GDRIVE_FILE_ID ontbreekt. Zet dit in Streamlit Secrets of upload het bestand handmatig.")

    # 2) Fallback: upload in de app
    st.subheader("Handmatige upload (fallback)")
    up = st.file_uploader("Upload TotaleTimetable_date.parquet", type=["parquet"])
    if up is None:
        st.stop()

    # schrijf naar FILE
    with open(FILE, "wb") as f:
        f.write(up.getbuffer())

    st.success("Bestand geüpload en opgeslagen. De app herstart nu.")
    st.rerun()


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
    dataset = ds.dataset(FILE, format="parquet")
    cols = ["Timestamp", signal]

    rng = np.random.default_rng(12345)
    reservoir_ts = []
    reservoir_val = []
    seen = 0

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


def make_line_figure(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line={"width": 2},
        name=title
    ))
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
        hovermode="x",
        showlegend=False,
    )
    return fig


def make_preview_figure_with_window(preview_df: pd.DataFrame, signal: str, start_dt: datetime, end_dt: datetime):
    fig = make_line_figure(preview_df["Timestamp"], preview_df[signal], f"Preview: {signal}")

    # Shaded selection + borders
    fig.add_vrect(
        x0=pd.Timestamp(start_dt),
        x1=pd.Timestamp(end_dt),
        fillcolor="rgba(0,0,0,0.12)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=pd.Timestamp(start_dt), line_width=2)
    fig.add_vline(x=pd.Timestamp(end_dt), line_width=2)

    fig.update_layout(title=f"Preview: {signal}  (selectie: {start_dt} → {end_dt})")
    return fig


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

sig1_default = choose_default(sig_candidates, ["EEC1_Speed"])
sig2_default = choose_default(sig_candidates, ["Verbruik_g_per_km"])
sig3_default = choose_default(sig_candidates, ["GPS_speed"])

# ---- tijdvenster UI eerst, zodat preview de selectie kan tonen
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
    st.time_input("Start tijd", key="start_time", step=60, on_change=update_from_inputs)
with cB:
    st.date_input("Eind datum", key="end_date", on_change=update_from_inputs)
    st.time_input("Eind tijd", key="end_time", step=60, on_change=update_from_inputs)

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
duration = pd.Timestamp(end_dt) - pd.Timestamp(start_dt)

# ---- preview met selectiezone
st.divider()
st.subheader("Preview (volledig signaal + selectiezone zichtbaar)")

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
    st.caption("De grijze zone toont het geselecteerde tijdslot (sliders/tijdvakken).")
    st.plotly_chart(
        make_preview_figure_with_window(preview_df, preview_signal, start_dt, end_dt),
        width="stretch",
    )

# ---- signalen kiezen voor export (en later visualisatie)
st.divider()
st.subheader("Signalen kiezen voor download")

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
    chosen = st.selectbox(
        f"Signaal {i+1} (typ om te zoeken)",
        sig_candidates,
        index=safe_index(sig_candidates, st.session_state.get(key, default_i), default_i),
        key=key,
    )
    selected_signals.append(chosen)

if len(set(selected_signals)) != len(selected_signals):
    st.warning("Kies elk signaal maar één keer (geen duplicaten).")
    st.stop()

# ✅ export bevat enkel: Timestamp + GPS + geselecteerde signalen
cols_needed = ["Timestamp", LON_COL, LAT_COL] + selected_signals

st.subheader("Download (CSV)")

start = pd.Timestamp(start_dt)
end = pd.Timestamp(end_dt)
start_str = start.strftime("%Y%m%d_%H%M%S")
end_str = end.strftime("%Y%m%d_%H%M%S")

if duration > BIG_WINDOW:
    st.warning(
        "Je geselecteerde tijdslot is groter dan 1 uur. "
        "Standaard kan je eerst enkel het eerste uur downloaden. "
        "Bevestig om het volledige tijdslot te downloaden."
    )

    first_end = min(end, start + BIG_WINDOW)
    export_first = load_filtered(cols_needed, start, first_end)
    st.download_button(
        "⬇️ Download eerste uur (veilig)",
        data=export_first.to_csv(index=False).encode("utf-8"),
        file_name=f"selectie_EERSTE_UUR_{start_str}_tot_{first_end.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.checkbox(
        "Ik begrijp dat >1 uur zwaar kan zijn en wil toch volledige download",
        key="confirm_big_export",
    )
    if st.session_state.get("confirm_big_export", False):
        export_full = load_filtered(cols_needed, start, end)
        st.download_button(
            "⬇️ Download volledige selectie (kan zwaar zijn)",
            data=export_full.to_csv(index=False).encode("utf-8"),
            file_name=f"selectie_VOLLEDIG_{start_str}_tot_{end_str}.csv",
            mime="text/csv",
        )
else:
    export_df = load_filtered(cols_needed, start, end)
    st.download_button(
        "⬇️ Download geselecteerd tijdvenster (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"selectie_{start_str}_tot_{end_str}.csv",
        mime="text/csv",
    )

# (visualisatie-deel kan je hierna terug toevoegen; jij vroeg vooral preview+selectie+download)
