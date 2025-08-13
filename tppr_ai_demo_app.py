
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
from pathlib import Path
import datetime as dt
import random
import io
import json
# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="OBW — TPPR AI Safety Assistant", layout="wide")

OBW_LOGO = Path("assets/obw_logo.png")

# -----------------------------------------------------------------------------
# STATIC CONFIG
# -----------------------------------------------------------------------------
ROOMS = [
    "Boiler Room",
    "Control Room",
    "Chemical Storage",
    "Maintenance Workshop",
    "Air Handling Unit",
    "Pharma Lab A",
    "Pharma Lab B",
    "Quality Control Lab",
    "Cleanroom",
    "Corridor North",
    "Corridor South",
]

PRIMARY_ROOMS = ROOMS[:]  # detectors in all for demo

DETECTOR_MODEL = {
    "Boiler Room": "Honeywell XNX",
    "Control Room": "Sensepoint XCD",
    "Chemical Storage": "Honeywell Midas",
    "Maintenance Workshop": "Honeywell XNX",
    "Air Handling Unit": "Sensepoint XCD",
    "Pharma Lab A": "Honeywell Midas",
    "Pharma Lab B": "Honeywell Midas",
    "Quality Control Lab": "Honeywell Midas",
    "Cleanroom": "Honeywell Midas",
    "Corridor North": "Sensepoint XCD",
    "Corridor South": "Sensepoint XCD",
}

GASES = ["Methane", "CO", "H2S"]

THRESHOLDS = {
    "Methane": {"warning": 50, "critical": 100},  # ppm (demo)
    "CO": {"warning": 25, "critical": 50},        # ppm
    "H2S": {"warning": 10, "critical": 20},       # ppm
}

ROOM_GAS_PROFILE = {
    "Boiler Room": ["Methane", "CO"],
    "Control Room": ["CO"],
    "Chemical Storage": ["H2S", "CO"],
    "Maintenance Workshop": ["CO"],
    "Air Handling Unit": ["CO"],
    "Pharma Lab A": ["H2S", "CO"],
    "Pharma Lab B": ["H2S", "CO"],
    "Quality Control Lab": ["H2S"],
    "Cleanroom": ["CO"],
    "Corridor North": ["CO"],
    "Corridor South": ["CO"],
}

# Facility vector geometry (0..100)
ROOM_RECTS = {
    # Top band (labs)
    "Pharma Lab A": (2, 68, 30, 96),
    "Pharma Lab B": (32, 68, 60, 96),
    "Quality Control Lab": (62, 68, 98, 96),

    # Middle band
    "Corridor North": (2, 58, 98, 66),
    "Air Handling Unit": (2, 34, 22, 56),
    "Chemical Storage": (24, 34, 44, 56),
    "Cleanroom": (46, 34, 70, 56),
    "Control Room": (72, 34, 98, 56),

    # Bottom band
    "Corridor South": (2, 26, 98, 32),
    "Boiler Room": (2, 2, 34, 24),
    "Maintenance Workshop": (36, 2, 72, 24),
}

HOTSPOTS = {
    "Pharma Lab A": (16, 82),
    "Pharma Lab B": (46, 82),
    "Quality Control Lab": (80, 82),
    "Corridor North": (50, 62),
    "Air Handling Unit": (12, 46),
    "Chemical Storage": (34, 46),
    "Cleanroom": (58, 46),
    "Control Room": (86, 46),
    "Corridor South": (50, 29),
    "Boiler Room": (18, 13),
    "Maintenance Workshop": (54, 13),
}

# Room-object layouts (vector rectangles with names & optional sensor mapping)
# Each object: dict(name, shape=(x0,y0,x1,y1), has_sensor:bool)
ROOM_OBJECTS = {
    "Boiler Room": [
        {"name": "Boiler 1 (XNX)", "shape": (6, 6, 16, 20), "has_sensor": True, "model": "Honeywell XNX", "gas": "Methane"},
        {"name": "Boiler 2 (XNX)", "shape": (20, 6, 30, 20), "has_sensor": True, "model": "Honeywell XNX", "gas": "CO"},
        {"name": "Vent Fan", "shape": (2, 10, 5, 16), "has_sensor": False},
    ],
    "Chemical Storage": [
        {"name": "Drum Rack A", "shape": (26, 38, 34, 44), "has_sensor": False},
        {"name": "Shelving B", "shape": (26, 46, 34, 52), "has_sensor": False},
        {"name": "Midas Point", "shape": (36, 40, 42, 46), "has_sensor": True, "model": "Honeywell Midas", "gas": "H2S"},
    ],
    "Pharma Lab A": [
        {"name": "Fume Hood 1", "shape": (4, 90, 12, 94), "has_sensor": False},
        {"name": "Bench A", "shape": (6, 74, 26, 78), "has_sensor": False},
        {"name": "Midas Point", "shape": (24, 86, 28, 90), "has_sensor": True, "model": "Honeywell Midas", "gas": "H2S"},
    ],
    "Pharma Lab B": [
        {"name": "Fume Hood 2", "shape": (34, 90, 42, 94), "has_sensor": False},
        {"name": "Bench B", "shape": (36, 74, 56, 78), "has_sensor": False},
        {"name": "Midas Point", "shape": (52, 86, 56, 90), "has_sensor": True, "model": "Honeywell Midas", "gas": "H2S"},
    ],
    "Quality Control Lab": [
        {"name": "QC Bench", "shape": (66, 74, 90, 78), "has_sensor": False},
        {"name": "Sample Store", "shape": (90, 74, 96, 86), "has_sensor": False},
        {"name": "Midas Point", "shape": (78, 88, 82, 92), "has_sensor": True, "model": "Honeywell Midas", "gas": "H2S"},
    ],
    "Cleanroom": [
        {"name": "Process Tool", "shape": (50, 40, 62, 50), "has_sensor": False},
        {"name": "Midas Point", "shape": (64, 48, 68, 52), "has_sensor": True, "model": "Honeywell Midas", "gas": "CO"},
    ],
    "Control Room": [
        {"name": "Operator Desk", "shape": (76, 50, 94, 54), "has_sensor": False},
        {"name": "Rack", "shape": (88, 36, 94, 44), "has_sensor": False},
        {"name": "Sensepoint", "shape": (82, 44, 86, 48), "has_sensor": True, "model": "Sensepoint XCD", "gas": "CO"},
    ],
    "Air Handling Unit": [
        {"name": "AHU", "shape": (4, 44, 18, 52), "has_sensor": False},
        {"name": "Sensepoint", "shape": (18, 50, 20, 52), "has_sensor": True, "model": "Sensepoint XCD", "gas": "CO"},
    ],
    "Maintenance Workshop": [
        {"name": "Workbench", "shape": (40, 6, 60, 10), "has_sensor": False},
        {"name": "Compressor", "shape": (64, 8, 70, 14), "has_sensor": False},
        {"name": "XNX Point", "shape": (56, 16, 60, 20), "has_sensor": True, "model": "Honeywell XNX", "gas": "CO"},
    ],
    "Corridor North": [
        {"name": "Sensepoint", "shape": (48, 60, 52, 64), "has_sensor": True, "model": "Sensepoint XCD", "gas": "CO"},
    ],
    "Corridor South": [
        {"name": "Sensepoint", "shape": (48, 28, 52, 30), "has_sensor": True, "model": "Sensepoint XCD", "gas": "CO"},
    ],
}

SAFE_COLOR = "#26a269"
WARN_COLOR = "#f99b11"
CRIT_COLOR = "#cc0000"
NEUTRAL_LINE = "#1f2937"
ROOM_FILL = "#ffffff"
CORRIDOR_FILL = "rgba(150,150,150,0.15)"
EXIT_COLOR = "#2ecc71"

# Doors: list of dicts with location & type
FIRE_EXITS = [(2, 66, 6, 68), (94, 26, 98, 28)]
NORMAL_DOORS = [
    (30, 68, 32, 66),   # between Pharma A & corridor north
    (60, 68, 62, 66),   # between Pharma B & corridor north
    (2, 56, 4, 58),     # AHU to corridor north
    (44, 56, 46, 58),   # Chem to corridor north
    (70, 56, 72, 58),   # Cleanroom to corridor north
    (98, 56, 96, 58),   # Control to corridor north
    (34, 26, 36, 32),   # Maint to corridor south
]

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
def init_data():
    now = pd.Timestamp.now().floor("min")
    ts = pd.date_range(now - pd.Timedelta(minutes=179), periods=180, freq="1min")
    rows = []
    rng = np.random.default_rng(42)
    for room in PRIMARY_ROOMS:
        for gas in GASES:
            base = {"Methane": 25, "CO": 15, "H2S": 6}[gas]
            noise = np.clip(rng.normal(0, 4, size=len(ts)), -7, 8)
            vals = np.clip(base + noise, 0, None)
            for t, v in zip(ts, vals):
                rows.append({"timestamp": t, "room": room, "gas": gas, "ppm": float(v)})
    return pd.DataFrame(rows)

if "view" not in st.session_state:
    st.session_state.view = "facility"
if "selected_room" not in st.session_state:
    st.session_state.selected_room = None
if "selected_equipment" not in st.session_state:
    st.session_state.selected_equipment = None
if "df" not in st.session_state:
    st.session_state.df = init_data()
if "incidents" not in st.session_state:
    st.session_state.incidents = []
if "last_simulated" not in st.session_state:
    st.session_state.last_simulated = None
if "evac_logs" not in st.session_state:
    st.session_state.evac_logs = []  # store route histories per event

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def logo():
    if OBW_LOGO.exists():
        st.image(str(OBW_LOGO), width=120)

def status_of(gas: str, ppm: float) -> str:
    th = THRESHOLDS[gas]
    if ppm >= th["critical"]:
        return "critical"
    if ppm >= th["warning"]:
        return "warning"
    return "safe"

def status_color(status: str) -> str:
    return {"safe": SAFE_COLOR, "warning": WARN_COLOR, "critical": CRIT_COLOR}[status]

def latest_snapshot_for_room(df: pd.DataFrame, room: str):
    r = df[df["room"] == room]
    if r.empty:
        return None
    tmax = r["timestamp"].max()
    snap = r[r["timestamp"] == tmax]
    vals = {}
    for g in GASES:
        row = snap[snap["gas"] == g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        vals[g] = (ppm, status_of(g, ppm))
    return tmax, vals

def overall_status(vals_dict) -> str:
    if any(s == "critical" for (_, s) in vals_dict.values()):
        return "critical"
    if any(s == "warning" for (_, s) in vals_dict.values()):
        return "warning"
    return "safe"

def linear_predict(df_room_gas: pd.DataFrame, minutes_ahead=20):
    if df_room_gas.empty:
        return None
    tail = df_room_gas.sort_values("timestamp").tail(12).copy()
    if len(tail) < 3:
        return None
    t0 = tail["timestamp"].min()
    tail["m"] = (tail["timestamp"] - t0).dt.total_seconds() / 60.0
    x = tail["m"].values
    y = tail["ppm"].values
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    last_m = x[-1]
    fut_m = np.arange(last_m + 1, last_m + minutes_ahead + 1)
    fut_ts = [tail["timestamp"].iloc[-1] + pd.Timedelta(minutes=int(k - last_m)) for k in fut_m]
    fut_y = m * fut_m + c
    return pd.DataFrame({"timestamp": fut_ts, "ppm": fut_y})

def time_to_next_threshold(df_room_gas: pd.DataFrame, gas: str):
    pred = linear_predict(df_room_gas, minutes_ahead=30)
    if pred is None or pred.empty:
        return None
    for level in ["warning", "critical"]:
        th = THRESHOLDS[gas][level]
        hit = pred[pred["ppm"] >= th]
        if not hit.empty:
            eta = hit["timestamp"].iloc[0] - df_room_gas["timestamp"].max()
            return level, eta
    return None

def ai_room_summary(room: str, df_room: pd.DataFrame) -> str:
    if df_room.empty:
        return "No data available."
    last_ts = df_room["timestamp"].max()
    snap = df_room[df_room["timestamp"] == last_ts]
    lines = [f"AI Safety Summary — {room}", f"Time: {last_ts.strftime('%Y-%m-%d %H:%M')}"]
    for g in GASES:
        row = snap[snap["gas"] == g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        stt = status_of(g, ppm)
        if room == "Boiler Room" and g == "Methane" and stt != "safe":
            advice = "Shut burners, isolate ignition sources, ventilate, evacuate."
        elif room == "Chemical Storage" and g == "H2S" and stt != "safe":
            advice = "Seal room, don SCBA, isolate ventilation, alert hazmat."
        elif room in ["Pharma Lab A", "Pharma Lab B", "Quality Control Lab"] and stt != "safe":
            advice = "Stop experiments, evacuate, start fume hood purge, investigate."
        elif room == "Cleanroom" and stt != "safe":
            advice = "Evacuate, initiate cleanroom purge, check make-up air."
        else:
            advice = ("Levels normal. Continue monitoring & verify calibration."
                      if stt == "safe" else
                      "Increase ventilation; begin source hunt; monitor closely.")
        lines.append(f"{g}: {ppm:.0f} ppm — {stt.upper()}. {advice}")
    lines.append("\nPreventative measures:")
    lines.append("• Keep calibration & bump-test schedule up to date.")
    lines.append("• Verify extraction / make-up air performance.")
    lines.append("• Review near-misses; update SOPs & training.")
    return "\n".join(lines)

def safety_manual_text(room: str, model: str, gas: str) -> str:
    steps = [
        f"Device: {model} — Location: {room}",
        "1) Confirm reading on secondary display/logs.",
        "2) If WARNING: increase ventilation, start source hunt, brief staff.",
        "3) If CRITICAL: evacuate affected area, isolate ignition sources.",
        f"4) Gas-specific PPE: {'SCBA' if gas=='H2S' else 'Appropriate respiratory protection'}.",
        "5) Verify with fixed/portable instrument (bump nearby).",
        "6) After control: isolate leak, RCA, reset & re-arm detector.",
    ]
    return "\n".join(steps)

def simulate_event(df: pd.DataFrame):
    room = random.choice(PRIMARY_ROOMS)
    gas = random.choice(ROOM_GAS_PROFILE.get(room, GASES))
    now = df["timestamp"].max()
    base = {"Methane": (110, 180), "CO": (55, 120), "H2S": (22, 60)}[gas]
    for i in range(6):
        t = now - pd.Timedelta(minutes=i)
        idx = (df["timestamp"] == t) & (df["room"] == room) & (df["gas"] == gas)
        if idx.any():
            df.loc[idx, "ppm"] = random.randint(*base)
    st.session_state.df = df
    incident = {
        "id": f"INC-{int(dt.datetime.utcnow().timestamp())}-{random.randint(100,999)}",
        "time": now.strftime("%Y-%m-%d %H:%M"),
        "room": room,
        "gas": gas,
        "ppm": int(df[(df['timestamp']==now)&(df['room']==room)&(df['gas']==gas)]['ppm'].iloc[0]),
        "level": "Critical",
        "note": "Simulated spike"
    }
    st.session_state.incidents.insert(0, incident)
    st.session_state.last_simulated = incident

def replay_last_incident():
    if not st.session_state.last_simulated:
        st.warning("No incident to replay yet.")
        return
    inc = st.session_state.last_simulated
    df = st.session_state.df
    room, gas = inc["room"], inc["gas"]
    now = df["timestamp"].max()
    mask = (df["timestamp"]>= now - pd.Timedelta(minutes=7)) & (df["room"]==room) & (df["gas"]==gas)
    df = df[~mask].copy()
    times = pd.date_range(now - pd.Timedelta(minutes=7), periods=8, freq="1min")
    vals = np.linspace(THRESHOLDS[gas]["warning"]-5, THRESHOLDS[gas]["critical"]+30, num=8)
    for t, v in zip(times, vals):
        df.loc[len(df)] = {"timestamp": t, "room": room, "gas": gas, "ppm": float(v)}
    st.session_state.df = df.sort_values("timestamp")

# -----------------------------------------------------------------------------
# FACILITY FIGURE
# -----------------------------------------------------------------------------
def build_facility_figure(df: pd.DataFrame):
    fig = go.Figure()

    # Rooms
    for room, (x0, y0, x1, y1) in ROOM_RECTS.items():
        fill = CORRIDOR_FILL if room.startswith("Corridor") else ROOM_FILL
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=NEUTRAL_LINE, width=2),
                      fillcolor=fill)
        # Room label
        fig.add_annotation(x=x0 + 2, y=y1 - 3, text=room, showarrow=False,
                           font=dict(size=12, color="#111"), xanchor="left")

    # Stairs block
    fig.add_shape(type="rect", x0=74, y0=2, x1=98, y1=24,
                  line=dict(color=NEUTRAL_LINE, width=2), fillcolor="#f2f2f2")
    fig.add_annotation(x=86, y=13, text="Stairs", showarrow=False, font=dict(size=12))

    # Doors
    for (x0,y0,x1,y1) in NORMAL_DOORS:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#777", width=1), fillcolor="#eee")
    for (x0,y0,x1,y1) in FIRE_EXITS:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#178f4a", width=1), fillcolor=EXIT_COLOR)
        fig.add_annotation(x=(x0+x1)/2, y=y1+1.5, text="EXIT", showarrow=False,
                           font=dict(size=10, color="#0b5f31"))

    # Detector dots
    xs, ys, sizes, colors, texts, custom = [], [], [], [], [], []
    for room in PRIMARY_ROOMS:
        tvals = latest_snapshot_for_room(df, room)
        if tvals:
            _, vals = tvals
            stt = overall_status(vals)
        else:
            stt = "safe"
            vals = {g: (0.0, "safe") for g in GASES}

        color = status_color(stt)
        size = 18 if stt == "safe" else (20 if stt == "warning" else 24)

        hover_lines = [f"<b>{room}</b> — {DETECTOR_MODEL[room]}"]
        for g in GASES:
            ppm, s = vals[g]
            hover_lines.append(f"{g}: {ppm:.0f} ppm ({s})")
        x, y = HOTSPOTS[room]
        # Fix Corridor North label position (slightly lower)
        label_y = y - 5 if room != "Corridor North" else y - 7

        fig.add_annotation(x=x, y=label_y, text=DETECTOR_MODEL[room],
                           showarrow=False, font=dict(size=10, color="#222"),
                           xanchor="center")

        xs.append(x); ys.append(y); sizes.append(size); colors.append(color); custom.append(json.dumps({"type":"detector","room":room}))
        texts.append("<br>".join(hover_lines))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="#ffffff")),
        hoverinfo="text", text=texts, customdata=custom, name="Detectors"
    ))

    # Clickable exits overlay (as points)
    ex_x, ex_y, ex_cust = [], [], []
    for (x0,y0,x1,y1) in FIRE_EXITS:
        cx, cy = (x0+x1)/2, (y0+y1)/2
        ex_x.append(cx); ex_y.append(cy); ex_cust.append(json.dumps({"type":"exit"}))
    fig.add_trace(go.Scatter(
        x=ex_x, y=ex_y, mode="markers",
        marker=dict(size=10, color=EXIT_COLOR, symbol="square"),
        hoverinfo="none", showlegend=False, customdata=ex_cust
    ))

    fig.update_xaxes(range=[0,100], visible=False)
    fig.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=620)
    return fig

def facility_controls():
    c0, c1, c2, c3, c4 = st.columns([1.4, 2.4, 2.2, 1.6, 1.2])
    with c0:
        logo()
        st.markdown("### OBW — TPPR AI")
    with c1:
        st.markdown("**Legend**")
        a,b,c = st.columns(3)
        with a: st.markdown(f"<div style='color:{SAFE_COLOR}'>● Safe</div>", unsafe_allow_html=True)
        with b: st.markdown(f"<div style='color:{WARN_COLOR}'>● Warning</div>", unsafe_allow_html=True)
        with c: st.markdown(f"<div style='color:{CRIT_COLOR}'>● Critical</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Actions**")
        if st.button("Simulate Live Gas Event", use_container_width=True):
            simulate_event(st.session_state.df)
        if st.button("Replay Last Incident", use_container_width=True):
            replay_last_incident()
        if st.button("View Evacuation Plan", use_container_width=True):
            st.session_state.view = "evac"
    with c3:
        df = st.session_state.df
        alert_txt = "All clear — no current alerts."
        for room in PRIMARY_ROOMS:
            snap = latest_snapshot_for_room(df, room)
            if not snap: continue
            _, vals = snap
            stt = overall_status(vals)
            if stt in ("warning", "critical"):
                worst = max(GASES, key=lambda g: {"safe":0,"warning":1,"critical":2}[vals[g][1]])
                ppm = vals[worst][0]
                alert_txt = f"**Latest Alert:** {room} — {worst} {ppm:.0f} ppm ({stt.upper()})"
                break
        st.info(alert_txt)
    with c4:
        st.markdown("**Reports**")
        if st.session_state.incidents:
            csv = pd.DataFrame(st.session_state.incidents).to_csv(index=False).encode("utf-8")
            st.download_button("Download Incidents CSV", data=csv, file_name="incidents.csv", mime="text/csv", use_container_width=True)
        else:
            st.caption("No incidents yet.")

def render_facility():
    facility_controls()
    fig = build_facility_figure(st.session_state.df)

    clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=640, override_width="100%")
    if clicked and len(clicked) > 0 and "customdata" in clicked[0]:
        info = json.loads(clicked[0]["customdata"])
        if info.get("type") == "detector":
            st.session_state.selected_room = info["room"]
            st.session_state.view = "room"
        elif info.get("type") == "exit":
            st.session_state.view = "evac"

    st.markdown("---")
    st.markdown("#### Incident History")
    if not st.session_state.incidents:
        st.caption("No incidents recorded.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.incidents), use_container_width=True)

# -----------------------------------------------------------------------------
# ROOM VIEW (VECTOR WITH OBJECTS & DOORS)
# -----------------------------------------------------------------------------
def build_room_diagram(room: str):
    x0,y0,x1,y1 = ROOM_RECTS.get(room, (10,10,90,90))
    w = x1-x0; h=y1-y0
    R = lambda rx,ry: (x0 + rx, y0 + ry)  # relative mapper

    fig = go.Figure()
    # Room boundary
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                  line=dict(color=NEUTRAL_LINE, width=3), fillcolor="#fafafa")
    # Doors facing corridor (rough heuristic: place at center of the wall touching corridor)
    # We’ll just draw a couple of door icons:
    fig.add_shape(type="rect", x0=x0+(w*0.45), y0=y0-1.5, x1=x0+(w*0.55), y1=y0,
                  line=dict(color="#777", width=1), fillcolor="#eee")  # top door
    fig.add_annotation(x=x0+(w*0.5), y=y0-2.5, text="Door (click to go back)", showarrow=False, font=dict(size=10, color="#666"))

    # Objects
    objs = ROOM_OBJECTS.get(room, [])
    obj_x, obj_y, obj_text, obj_cust = [], [], [], []
    for o in objs:
        sx0, sy0, sx1, sy1 = o["shape"]
        fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1,
                      line=dict(color="#888", width=2), fillcolor="#e9eef3")
        cx, cy = (sx0+sx1)/2, (sy0+sy1)/2
        obj_x.append(cx); obj_y.append(cy)
        obj_text.append(o["name"])
        obj_cust.append(json.dumps({"type":"object","room":room,"name":o["name"],"has_sensor":o.get("has_sensor",False),"model":o.get("model"),"gas":o.get("gas")}))
        fig.add_annotation(x=cx, y=sy1+1.2, text=o["name"], showarrow=False, font=dict(size=10, color="#333"), xanchor="center")

    # Detector dot in room (use HOTSPOTS room position)
    if room in HOTSPOTS:
        dx, dy = HOTSPOTS[room]
        fig.add_trace(go.Scatter(
            x=[dx], y=[dy], mode="markers",
            marker=dict(size=18, color=SAFE_COLOR, line=dict(width=2, color="#fff")),
            hoverinfo="text",
            text=[f"{room} — {DETECTOR_MODEL.get(room,'Honeywell Detector')}"],
            customdata=[json.dumps({"type":"room_detector","room":room})],
            name="Room Detector"
        ))

    # Object clickable points
    if obj_x:
        fig.add_trace(go.Scatter(
            x=obj_x, y=obj_y, mode="markers",
            marker=dict(size=12, color="#6b7280"),
            hoverinfo="text", text=obj_text, customdata=obj_cust,
            name="Equipment"
        ))

    # Fire exit icon inside room (optional visual)
    # (not used for click nav here; evacuation handled via facility)
    fig.update_xaxes(range=[x0-2,x1+2], visible=False)
    fig.update_yaxes(range=[y0-6,y1+2], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
    return fig

def render_room(room: str):
    top = st.columns([1,3,2])
    with top[0]:
        logo()
    with top[1]:
        st.markdown(f"### {room} — {DETECTOR_MODEL.get(room, 'Honeywell Detector')}")
    with top[2]:
        c = st.columns(2)
        with c[0]:
            if st.button("Back to Facility", use_container_width=True):
                st.session_state.view = "facility"
        with c[1]:
            if st.button("Evacuation Plan", use_container_width=True):
                st.session_state.view = "evac"

    fig = build_room_diagram(room)
    clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=440, override_width="100%")

    # Handle clicks in room view
    if clicked and len(clicked) > 0 and "customdata" in clicked[0]:
        info = json.loads(clicked[0]["customdata"])
        if info.get("type") == "object" and info.get("has_sensor"):
            st.session_state.selected_equipment = info
            st.session_state.view = "equipment"
        elif info.get("type") == "room_detector":
            # open the room detector (aggregate) as equipment view too
            st.session_state.selected_equipment = {"type":"object","room":room,"name":"Room Detector","has_sensor":True,"model":DETECTOR_MODEL.get(room,"Honeywell"),"gas":None}
            st.session_state.view = "equipment"

    # Below: live chart + AI summary
    df = st.session_state.df
    df_room = df[df["room"] == room].copy().sort_values("timestamp")
    c1, c2 = st.columns([3, 2])
    with c1:
        if df_room.empty:
            st.warning("No data for this room.")
        else:
            fig2 = px.line(df_room, x="timestamp", y="ppm", color="gas", title="Live Readings")
            for g in GASES:
                fig2.add_hline(y=THRESHOLDS[g]["warning"], line=dict(color=WARN_COLOR, dash="dot"))
                fig2.add_hline(y=THRESHOLDS[g]["critical"], line=dict(color=CRIT_COLOR, dash="dot"))
            fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380, legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown("**AI Assistant — Room Summary**")
        st.text_area("", ai_room_summary(room, df_room), height=340)

# -----------------------------------------------------------------------------
# EQUIPMENT / SENSOR VIEW
# -----------------------------------------------------------------------------
def render_equipment(equip: dict):
    room = equip["room"]
    name = equip["name"]
    model = equip.get("model", DETECTOR_MODEL.get(room, "Honeywell"))
    gas_pref = equip.get("gas")  # may be None for room detector

    top = st.columns([1,3,2])
    with top[0]:
        logo()
    with top[1]:
        st.markdown(f"### {room} → {name} — {model}")
    with top[2]:
        a,b,c = st.columns(3)
        with a:
            if st.button("Back to Room", use_container_width=True):
                st.session_state.view = "room"
        with b:
            if st.button("Replay Last Incident", use_container_width=True):
                replay_last_incident()
        with c:
            if st.button("Evacuation Plan", use_container_width=True):
                st.session_state.view = "evac"

    df = st.session_state.df
    df_room = df[df["room"] == room].copy()
    if gas_pref:
        gases = [gas_pref]
    else:
        gases = GASES

    c1, c2 = st.columns([3,2])
    with c1:
        if df_room.empty:
            st.warning("No data for this equipment.")
        else:
            fig = go.Figure()
            for g in gases:
                dfg = df_room[df_room["gas"] == g].sort_values("timestamp")
                fig.add_trace(go.Scatter(x=dfg["timestamp"], y=dfg["ppm"], mode="lines", name=g))
                pred = linear_predict(dfg, 15)
                if pred is not None:
                    fig.add_trace(go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name=f"{g} — forecast", line=dict(dash="dash")))
                fig.add_hline(y=THRESHOLDS[g]["warning"], line=dict(color=WARN_COLOR, dash="dot"))
                fig.add_hline(y=THRESHOLDS[g]["critical"], line=dict(color=CRIT_COLOR, dash="dot"))
            fig.update_layout(title="Live Readings + 15‑min Forecast", margin=dict(l=10, r=10, t=50, b=10), height=420, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**AI Assistant — Equipment Guidance**")
        # Build explanation per selected gases
        msg_lines = [f"Equipment: {name} ({model}) — Room: {room}"]
        snap_ts = df_room["timestamp"].max() if not df_room.empty else None
        if snap_ts is not None:
            msg_lines.append(f"Time: {snap_ts.strftime('%Y-%m-%d %H:%M')}")
        for g in gases:
            dfg = df_room[df_room["gas"] == g]
            if dfg.empty: continue
            last_ppm = float(dfg.sort_values('timestamp')["ppm"].iloc[-1])
            stt = status_of(g, last_ppm)
            eta = time_to_next_threshold(dfg, g)
            eta_txt = ""
            if eta:
                level, delta = eta
                mins = int(max(delta.total_seconds()/60, 0))
                eta_txt = f" (projected {level.upper()} in ~{mins} min)"
            msg_lines.append(f"{g}: {last_ppm:.0f} ppm — {stt.upper()}{eta_txt}")
        msg_lines.append("\nRecommendations:")
        if room == "Boiler Room":
            msg_lines.append("• Isolate ignition, shut burners, ventilate. Verify gas train.")
        elif room == "Chemical Storage":
            msg_lines.append("• Seal area, don SCBA, isolate ventilation, alert hazmat.")
        else:
            msg_lines.append("• Increase ventilation, begin source hunt, brief staff.")
        st.text_area("", "\n".join(msg_lines), height=360)

    # Export room data & Safety Manual
    st.markdown("---")
    b1,b2,b3 = st.columns([1,1,2])
    with b1:
        if not df_room.empty:
            csv = df_room.sort_values("timestamp").to_csv(index=False).encode("utf-8")
            st.download_button("Download Room Data (CSV)", data=csv, file_name=f"{room.replace(' ','_')}_readings.csv", mime="text/csv", use_container_width=True)
    with b2:
        g_for_manual = gas_pref or max(GASES, key=lambda g: df_room[df_room["gas"]==g]["ppm"].iloc[-1] if not df_room[df_room["gas"]==g].empty else -1)
        manual = safety_manual_text(room, model, g_for_manual)
        st.download_button("View Safety Manual (HTML)", data=manual_html(room, model, manual).encode("utf-8"),
                           file_name=f"safety_manual_{room.replace(' ','_')}.html", mime="text/html", use_container_width=True)

# -----------------------------------------------------------------------------
# EVACUATION VIEW (DYNAMIC ROUTES + LOGGING)
# -----------------------------------------------------------------------------
def current_leak_rooms():
    """Heuristic: rooms with any gas at/over critical in last timestamp."""
    df = st.session_state.df
    tmax = df["timestamp"].max()
    snap = df[df["timestamp"] == tmax]
    leak_rooms = []
    for room in PRIMARY_ROOMS:
        rows = snap[snap["room"] == room]
        if rows.empty: continue
        if any(status_of(g, float(rows[rows['gas']==g]['ppm'].iloc[0] if not rows[rows['gas']==g].empty else 0.0)) == "critical" for g in GASES):
            leak_rooms.append(room)
    return leak_rooms

def simple_route(start_room: str):
    """Toy router: prefer Corridor North -> Exit 1; if CN unsafe, use Corridor South -> Exit 2."""
    blocked = set(current_leak_rooms())
    route = []
    if "Corridor North" in blocked:
        route = ["Corridor South", "Stairs", "Exit South"]
    else:
        route = ["Corridor North", "Exit North"]
    return route, blocked

def evacuation_html_steps(route, blocked):
    steps = ["<ol>"]
    steps.append("<li>Stay calm and follow nearest green route.</li>")
    if "Corridor North" in blocked:
        steps.append("<li>Corridor North unsafe — rerouting via Corridor South & Stairs.</li>")
    steps.append("<li>Proceed to designated assembly point. Account for personnel.</li>")
    steps.append("</ol>")
    return "\n".join(steps)

def render_evac():
    top = st.columns([1,3,2])
    with top[0]:
        logo()
    with top[1]:
        st.markdown("### Emergency Evacuation Plan")
    with top[2]:
        a,b,c = st.columns(3)
        with a:
            if st.button("Back to Facility", use_container_width=True):
                st.session_state.view = "facility"
        with b:
            if st.button("Simulate Live Gas Event", use_container_width=True):
                simulate_event(st.session_state.df)
        with c:
            if st.button("Replay Last Incident", use_container_width=True):
                replay_last_incident()

    # Build facility figure and overlay routes
    fig = build_facility_figure(st.session_state.df)

    # Find a "current room" context (use selected_room or default)
    start_room = st.session_state.selected_room or "Control Room"
    route, blocked = simple_route(start_room)

    # Draw route lines
    def room_center(r):
        x0,y0,x1,y1 = ROOM_RECTS[r]
        return (x0+x1)/2, (y0+y1)/2
    points = []
    # Start at current room center
    sx,sy = room_center(start_room)
    points.append((sx,sy))
    for step in route:
        if step == "Exit North":
            # use first fire exit
            (x0,y0,x1,y1) = FIRE_EXITS[0]
            points.append(((x0+x1)/2, (y0+y1)/2))
        elif step == "Exit South":
            (x0,y0,x1,y1) = FIRE_EXITS[1]
            points.append(((x0+x1)/2, (y0+y1)/2))
        elif step == "Stairs":
            points.append(((74+98)/2, (2+24)/2))
        else:
            points.append(room_center(step))

    xs, ys = zip(*points)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color=EXIT_COLOR, width=4),
                             marker=dict(size=6, color=EXIT_COLOR),
                             name="Evac Route"))

    # Highlight blocked areas
    for r in blocked:
        x0,y0,x1,y1 = ROOM_RECTS[r]
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=CRIT_COLOR, width=2), fillcolor="rgba(204,0,0,0.15)")

    st.plotly_chart(fig, use_container_width=True, height=640)

    # Log evacuation decision
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    log = {
        "time": now,
        "start_room": start_room,
        "blocked": list(blocked),
        "route": route,
        "status": "planned"
    }
    # Only append if changed from last
    if not st.session_state.evac_logs or st.session_state.evac_logs[0] != log:
        st.session_state.evac_logs.insert(0, log)

    # AI steps + downloads
    st.markdown("#### AI Evacuation Steps")
    st.markdown(evacuation_html_steps(route, blocked), unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        # Download evac log CSV
        if st.session_state.evac_logs:
            csv = pd.DataFrame(st.session_state.evac_logs).to_csv(index=False).encode("utf-8")
            st.download_button("Download Evacuation Logs (CSV)", data=csv, file_name="evac_logs.csv", mime="text/csv", use_container_width=True)
    with colB:
        # Download post-incident AI review (HTML)
        html = incident_review_html()
        st.download_button("Download Post‑Incident AI Review (HTML)", data=html.encode("utf-8"),
                           file_name="post_incident_review.html", mime="text/html", use_container_width=True)

# -----------------------------------------------------------------------------
# REPORT (HTML for print‑to‑PDF)
# -----------------------------------------------------------------------------
def manual_html(room: str, model: str, manual_text: str) -> str:
    logo_html = f"<img src='assets/obw_logo.png' style='height:48px'>" if OBW_LOGO.exists() else "<strong>OBW</strong>"
    return f"""
<html>
<head>
<meta charset="utf-8" />
<title>Safety Manual — {room}</title>
<style>
body {{ font-family: Arial, sans-serif; color:#111; }}
.header {{ display:flex; align-items:center; gap:12px; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:16px; margin-top:12px; }}
h1 {{ font-size:20px; margin:6px 0; }}
pre {{ white-space:pre-wrap; }}
</style>
</head>
<body>
<div class="header">{logo_html}<h1>Safety Manual — {room} — {model}</h1></div>
<div class="card"><pre>{manual_text}</pre></div>
</body></html>
"""

def incident_review_html() -> str:
    logo_html = f"<img src='assets/obw_logo.png' style='height:48px'>" if OBW_LOGO.exists() else "<strong>OBW</strong>"
    # Build summary from latest incident + evac logs
    inc = st.session_state.last_simulated or (st.session_state.incidents[0] if st.session_state.incidents else None)
    evac = st.session_state.evac_logs[0] if st.session_state.evac_logs else None
    when = inc["time"] if inc else dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    room = inc["room"] if inc else (st.session_state.selected_room or "—")
    gas = inc["gas"] if inc else "—"
    level = inc["level"] if inc else "—"
    score = 92 if evac and "Corridor North" in evac.get("blocked", []) else 88  # playful scoring

    steps = []
    steps.append(f"- Detection in <b>{room}</b> — Gas: <b>{gas}</b> — Level: <b>{level}</b>.")
    if evac:
        steps.append(f"- Initial route: {', '.join(evac['route'])}.")
        if "Corridor North" in evac.get("blocked", []):
            steps.append("- Reroute applied due to Corridor North unsafe.")
        steps.append("- Staff directed to nearest safe exit; assembly point confirmed.")

    suggestions = [
        "Increase ventilation capacity in affected corridor.",
        "Add secondary detector near likely leak source.",
        "Review evacuation drill frequency and signage clarity."
    ]

    return f"""
<html>
<head>
<meta charset="utf-8" />
<title>Post‑Incident AI Review</title>
<style>
body {{ font-family: Arial, sans-serif; color:#111; }}
.header {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
h1 {{ font-size:22px; margin:6px 0; }}
h2 {{ font-size:16px; margin:12px 0 6px; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0; }}
.badge {{ display:inline-block; background:#f2f2f2; padding:4px 8px; border-radius:6px; }}
ul {{ margin:0; padding-left:18px; }}
</style>
</head>
<body>
<div class="header">{logo_html}<div><h1>Post‑Incident AI Review</h1><div class="badge">Time: {when}</div></div></div>
<div class="card">
<h2>Summary</h2>
<ul>
{''.join([f"<li>{s}</li>" for s in steps])}
</ul>
</div>
<div class="card">
<h2>Response Quality</h2>
<p>Safety Response Score: <b>{score}/100</b></p>
<p>Evacuation started promptly; route{'s' if evac and len(evac.get('blocked',[]))>0 else ''} {'were updated dynamically' if evac and evac.get('blocked') else 'remained clear'}.</p>
</div>
<div class="card">
<h2>Risk Reduction Suggestions</h2>
<ul>
{''.join([f"<li>{s}</li>" for s in suggestions])}
</ul>
</div>
</body></html>
"""

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
if st.session_state.view == "facility":
    render_facility()
elif st.session_state.view == "room":
    render_room(st.session_state.selected_room or "Boiler Room")
elif st.session_state.view == "equipment":
    render_equipment(st.session_state.selected_equipment or {"room":"Boiler Room","name":"Room Detector","model":"Honeywell"})
else:
    render_evac()



