
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from pathlib import Path
from PIL import Image
import os, random

st.set_page_config(page_title="OBW | AI Safety Assistant", layout="wide")

# ---------- Styles ----------
st.markdown('''
<style>
:root {
  --bg:#0f172a; --panel:#111827; --text:#e5e7eb; --muted:#9ca3af;
  --ok:#22c55e; --warn:#f59e0b; --crit:#ef4444; --accent:#60a5fa;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--panel); }
.card { background: #0b1220; border: 1px solid #1f2937; padding: 14px; border-radius: 14px; }
.ai { border-left:4px solid var(--accent); }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px; margin-right:6px; background:#1f2937; border:1px solid #334155; }
.badge-warning { color:#8a4b00; border-color:#f59e0b; background:rgba(245,158,11,0.12); }
.badge-critical { color:#7a0000; border-color:#ef4444; background:rgba(239,68,68,0.12); }
.header { display:flex; align-items:center; gap:12px; }
.header img { height:38px; }
.header h3 { margin:0; }
</style>
''', unsafe_allow_html=True)

# ---------- Assets & Paths ----------
ASSETS_DIR = Path(__file__).parent / "assets"
FACILITY_IMG_PATH = ASSETS_DIR / "facility.png"
LOGO_IMG_PATH = ASSETS_DIR / "obw_logo.png"

# Load images safely (raise helpful message if missing)
if not FACILITY_IMG_PATH.exists() or not LOGO_IMG_PATH.exists():
    st.error("Assets missing. Ensure 'assets/facility.png' and 'assets/obw_logo.png' are in the repo.")
    st.stop()

facility_img = Image.open(FACILITY_IMG_PATH)
logo_img = Image.open(LOGO_IMG_PATH)

# ---------- Data ----------
THRESHOLDS = {
  "CH4":{"warning":25,"critical":50},
  "H2S":{"warning":10,"critical":20},
  "CO":{"warning":30,"critical":60},
}
GASES = ["CH4","H2S","CO"]

if "df" not in st.session_state:
    if os.path.exists("sample_data.csv"):
        df = pd.read_csv("sample_data.csv", parse_dates=["timestamp"])
    else:
        st.error("sample_data.csv not found in repo root.")
        st.stop()
    st.session_state.df = df
    st.session_state.selected_room = None
    st.session_state.incident_history = []

df = st.session_state.df

# ---------- Helper functions ----------
def status_label(g, v):
    thr = THRESHOLDS[g]
    if v >= thr["critical"]:
        return "critical"
    if v >= thr["warning"]:
        return "warning"
    return "safe"

def room_status(room_id):
    d = df[df["room"]==room_id].sort_values("timestamp")
    if d.empty:
        return "safe"
    latest = d.groupby("gas").tail(1)
    lvl = "safe"
    for _,row in latest.iterrows():
        s = status_label(row["gas"], row["ppm"])
        if s=="critical":
            return "critical"
        if s=="warning":
            lvl="warning"
    return lvl

ROOM_LABELS = {
    "1":"Upper Conveyor Hall",
    "2":"Loading Bay",
    "3":"Aux Room",
    "4":"Stairwell",
    "5":"Process Vessels",
    "6":"Main Floor",
    "7":"Piping Junction",
    "8":"Heat Exchangers",
    "9":"Compressor Skids",
    "10":"Discharge Area",
    "11":"Warehouse",
}

def build_incident_summary(room_id, when):
    names = ROOM_LABELS
    room_df = df[(df["room"]==room_id) & (df["timestamp"]<=when)].sort_values("timestamp")
    if room_df.empty or room_df["timestamp"].isna().all():
        return f"Incident — {names[room_id]}\nNo data available for this location."
    last_ts = room_df["timestamp"].max()
    snapshot = room_df[room_df["timestamp"]==last_ts]
    day_start = last_ts.floor("D")
    lines = [f"Incident — {names[room_id]}", f"Time: {last_ts.strftime('%Y-%m-%d %H:%M')}"]
    for g in GASES:
        gday = df[(df['room']==room_id)&(df['gas']==g)&(df['timestamp']>=day_start)&(df['timestamp']<=last_ts)]
        w = int((gday["ppm"] >= THRESHOLDS[g]["warning"]).sum())
        c = int((gday["ppm"] >= THRESHOLDS[g]["critical"]).sum())
        row = snapshot[snapshot["gas"]==g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        label = status_label(g, ppm).title()
        extra = f" — {c} critical event(s) today" if label=="Critical" else (f" — {w} warning event(s) today" if label=="Warning" else "")
        lines.append(f"{g}: {ppm} ppm ({label}){extra}")
    return "\n".join(lines)

# ---------- Header ----------
st.markdown(f'''
<div class="header">
  <img src="data:image/png;base64,{Path(LOGO_IMG_PATH).read_bytes().hex()}" alt="OBW"/>
  <h3>AI Safety Assistant</h3>
</div>
''', unsafe_allow_html=True)

# ---------- Facility Map (2.5D with Plotly background image) ----------
st.markdown("#### Facility Overview")

# Marker positions normalized (0..1) tuned for provided image
HOTSPOTS = {
    "1": (0.23, 0.18),
    "2": (0.08, 0.38),
    "3": (0.10, 0.62),
    "4": (0.33, 0.53),
    "5": (0.72, 0.07),
    "6": (0.54, 0.33),
    "7": (0.71, 0.40),
    "8": (0.87, 0.43),
    "9": (0.84, 0.18),
    "10": (0.58, 0.78),
    "11": (0.95, 0.92),
}

fig = go.Figure()
# Plotly supports PIL image as source; embed to avoid path issues
fig.update_layout(
    images=[dict(
        source=facility_img,
        xref="x", yref="y",
        x=0, y=1, sizex=1, sizey=1,
        sizing="stretch", layer="below"
    )],
    xaxis=dict(visible=False, range=[0,1]),
    yaxis=dict(visible=False, range=[0,1], scaleanchor="x", scaleratio=1),
    height=560, margin=dict(l=0,r=0,t=10,b=0)
)

xs, ys, colors, texts = [], [], [], []
for rid, (x,y) in HOTSPOTS.items():
    xs.append(x); ys.append(1-y)
    lvl = room_status(rid)
    col = {"safe":"#22c55e","warning":"#f59e0b","critical":"#ef4444"}[lvl]
    colors.append(col)
    d = df[df["room"]==rid].sort_values("timestamp").groupby("gas").tail(1)
    if d.empty:
        tip = f"{ROOM_LABELS[rid]} — No data"
    else:
        tip = f"{ROOM_LABELS[rid]} — " + ", ".join([f"{g} {round(v,1)} ppm" for g,v in zip(d['gas'], d['ppm'])])
    texts.append(tip)

fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers+text",
    marker=dict(size=26, color=colors, line=dict(color="#111827", width=2)),
    text=list(HOTSPOTS.keys()), textposition="middle center",
    hovertext=texts, hoverinfo="text"
))

# Use plotly_events to capture clicks (renders the chart itself)
clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560, key="facility_map")

# Handle selection
if clicks:
    idx = clicks[0].get("pointNumber", None)
    if idx is not None and 0 <= idx < len(HOTSPOTS):
        rid = list(HOTSPOTS.keys())[idx]
        st.session_state.selected_room = rid

# ---------- AI Assistant + Room Dashboard ----------
sel = st.session_state.selected_room
if sel:
    st.markdown("---")
    st.subheader(f"Location: {ROOM_LABELS[sel]} (#{sel})")

    # AI panel
    room_df = df[df["room"]==sel].sort_values("timestamp")
    latest = room_df.groupby("gas").tail(1).set_index("gas")["ppm"].to_dict() if not room_df.empty else {}
    msgs=[]; level="safe"
    for g in GASES:
        v = latest.get(g, 0.0)
        thr = THRESHOLDS[g]
        if v >= thr["critical"]:
            level="critical"; msgs.append(f"{g} at {v} ppm — critical.")
        elif v >= thr["warning"] and level!="critical":
            level="warning"; msgs.append(f"{g} at {v} ppm — warning.")
    if level=="safe":
        headline = f"✅ All clear — levels within safe range."
    elif level=="warning":
        headline = "⚠️ Warning — " + "; ".join(msgs)
    else:
        headline = "🔴 Critical — " + "; ".join(msgs) + " Immediate action required."
    st.markdown(f"<div class='card ai'>{headline}</div>", unsafe_allow_html=True)

    # Prediction summary
    if not room_df.empty:
        pred_lines = []
        for g in GASES:
            gdf = room_df[room_df["gas"]==g].tail(30)
            if len(gdf)>=3:
                x = np.arange(len(gdf)); y = gdf["ppm"].values.astype(float)
                m,b = np.polyfit(x,y,1)
                fut = m*(len(gdf)+5)+b
                pred_lines.append(f"{g} → ~{round(fut,1)} ppm in ~5 min")
        if pred_lines:
            st.caption("Prediction: " + " | ".join(pred_lines))

    # Copyable incident summary
    summary = build_incident_summary(sel, room_df["timestamp"].max() if not room_df.empty else pd.NaT)
    st.code(summary, language=None)

    colA, colB = st.columns([6,2])

    with colA:
        import plotly.express as px
        if not room_df.empty:
            recent = room_df.groupby("gas").tail(150)
            fig_line = px.line(recent, x="timestamp", y="ppm", color="gas",
                               color_discrete_map={"CH4":"#60a5fa","H2S":"#34d399","CO":"#f87171"},
                               title="Live Readings")
            # thresholds
            shapes=[]; annotations=[]
            for g, col in [("CH4","#60a5fa"), ("H2S","#34d399"), ("CO","#f87171")]:
                w=THRESHOLDS[g]["warning"]; c=THRESHOLDS[g]["critical"]
                shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=w, y1=w, line=dict(color="#f59e0b",width=2)))
                shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=c, y1=c, line=dict(color="#ef4444",width=2,dash="dash")))
                annotations.append(dict(x=1.0, xref="paper", y=w, yref="y", text=f"{g} Warn {w}", showarrow=False, xanchor="right", font=dict(size=10,color="#f59e0b")))
                annotations.append(dict(x=1.0, xref="paper", y=c, yref="y", text=f"{g} Crit {c}", showarrow=False, xanchor="right", font=dict(size=10,color="#ef4444")))
            fig_line.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0), legend_title_text="", shapes=shapes, annotations=annotations)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data available for this location yet. Use **Simulate Live Gas Event** to generate data.")

    with colB:
        st.image(logo_img, caption="OBW Technologies", use_column_width=True)
        if st.button("Back to Facility"):
            st.session_state.selected_room = None
            st.rerun()

# ---------- Simulate button ----------
st.markdown("---")
if st.button("Simulate Live Gas Event"):
    rid = random.choice(list(HOTSPOTS.keys()))
    gases = random.sample(GASES, k=random.choice([1,2,3]))
    severity = random.choice(["warning","critical"])
    peaks = {"CH4": 30 if severity=="warning" else 70,
             "H2S": 12 if severity=="warning" else 26,
             "CO":  40 if severity=="warning" else 80}
    dur = 8 if severity=="warning" else 14

    last_ts = st.session_state.df["timestamp"].max()
    start = last_ts + pd.Timedelta(minutes=1)
    new_rows=[]
    for i in range(dur):
        ts = start + pd.Timedelta(minutes=i)
        for room in list(HOTSPOTS.keys()):
            for g in GASES:
                base = st.session_state.df[(st.session_state.df["room"]==room)&(st.session_state.df["gas"]==g)].tail(1)["ppm"]
                val = float(base.iloc[0]) if len(base)>0 else 5.0
                if room==rid and g in gases:
                    # add spike
                    frac = i/max(dur-1,1)
                    bump = peaks[g]*(0.2+0.8*frac**1.4)
                    val = val + bump + np.random.normal(0,1.2)
                new_rows.append({"timestamp": ts, "room": room, "gas": g, "ppm": round(val,2)})
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)

    # log incident & auto-select
    st.session_state.selected_room = rid
    summary = build_incident_summary(rid, st.session_state.df["timestamp"].max())
    st.session_state.incident_history.insert(0, summary)
    st.rerun()

st.caption("Tip: Click numbers on the facility map to open each location. Use **Simulate** to generate incidents.")
