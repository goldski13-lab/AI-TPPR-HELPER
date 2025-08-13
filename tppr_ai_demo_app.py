
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

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="OBW — TPPR AI Safety Assistant", layout="wide")

# -----------------------------------------------------------------------------
# STATIC CONFIG
# -----------------------------------------------------------------------------
# Facility rooms (realistic names) and assigned Honeywell detector models
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

# Which rooms are "primary" (detectors live there). Corridors get a detector too for demo.
PRIMARY_ROOMS = [
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

# Gases monitored (keep three to keep UI clean)
GASES = ["Methane", "CO", "H2S"]

# Thresholds (approximate realistic demo values)
THRESHOLDS = {
    "Methane": {"warning": 50, "critical": 100},   # ppm (demo; real systems often %LEL—this keeps the demo simple)
    "CO": {"warning": 25, "critical": 50},         # ppm
    "H2S": {"warning": 10, "critical": 20},        # ppm
}

# Room type → gas profile (what’s most likely to spike there for simulation realism)
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

# Vector floor plan coordinates (0..100) — rooms are sized/placed to look architectural
# Each entry: (x0, y0, x1, y1)
ROOM_RECTS = {
    # Top band (labs)
    "Pharma Lab A": (2, 68, 30, 96),
    "Pharma Lab B": (32, 68, 60, 96),
    "Quality Control Lab": (62, 68, 98, 96),

    # Middle band (corridor north + rooms)
    "Corridor North": (2, 58, 98, 66),  # corridor strip
    "Air Handling Unit": (2, 34, 22, 56),
    "Chemical Storage": (24, 34, 44, 56),
    "Cleanroom": (46, 34, 70, 56),
    "Control Room": (72, 34, 98, 56),

    # Bottom band (corridor south + rooms)
    "Corridor South": (2, 26, 98, 32),
    "Boiler Room": (2, 2, 34, 24),
    "Maintenance Workshop": (36, 2, 72, 24),
    # stairs and fire exits added as overlays below
}

# Detector dot positions (roughly centered in each room)
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

SAFE_COLOR = "#26a269"
WARN_COLOR = "#f99b11"
CRIT_COLOR = "#cc0000"
NEUTRAL_LINE = "#1f2937"   # dark gray for walls
ROOM_FILL = "#ffffff"
CORRIDOR_FILL = "rgba(150,150,150,0.15)"
EXIT_COLOR = "#2ecc71"

# -----------------------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "facility"   # "facility" or "room"
if "selected_room" not in st.session_state:
    st.session_state.selected_room = None
if "df" not in st.session_state:
    # Create 90 minutes of demo data (1-min cadence)
    now = pd.Timestamp.now().floor("min")
    ts = pd.date_range(now - pd.Timedelta(minutes=89), periods=90, freq="1min")
    rows = []
    rng = np.random.default_rng()
    for room in PRIMARY_ROOMS:
        for gas in GASES:
            # Base levels by gas
            base = {"Methane": 25, "CO": 15, "H2S": 6}[gas]
            noise = np.clip(rng.normal(0, 4, size=len(ts)), -7, 7)
            vals = np.clip(base + noise, 0, None)
            for t, v in zip(ts, vals):
                rows.append({"timestamp": t, "room": room, "gas": gas, "ppm": float(v)})
    st.session_state.df = pd.DataFrame(rows)

if "incidents" not in st.session_state:
    st.session_state.incidents = []  # dicts: {id, time, room, gas, ppm, level, note}

if "last_simulated" not in st.session_state:
    st.session_state.last_simulated = None  # store for replay

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def status_of(gas: str, ppm: float) -> str:
    th = THRESHOLDS[gas]
    if ppm >= th["critical"]:
        return "critical"
    if ppm >= th["warning"]:
        return "warning"
    return "safe"

def status_color(status: str) -> str:
    return {"safe": SAFE_COLOR, "warning": WARN_COLOR, "critical": CRIT_COLOR}.get(status, SAFE_COLOR)

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
    # vals_dict: gas -> (ppm, status)
    if any(s == "critical" for (_, s) in vals_dict.values()):
        return "critical"
    if any(s == "warning" for (_, s) in vals_dict.values()):
        return "warning"
    return "safe"

def short_ai_expl(room: str, gas: str, ppm: float, status: str) -> str:
    room_type = room
    if status == "critical":
        if room_type == "Boiler Room" and gas == "Methane":
            return "CRITICAL: Possible gas leak — shut burners, ventilate, evacuate."
        if room_type == "Chemical Storage" and gas == "H2S":
            return "CRITICAL: Toxic gas — seal room, isolate ventilation, call hazmat."
        if room_type == "Cleanroom" and gas == "CO":
            return "CRITICAL: CO rising — evacuate, activate purge, investigate source."
        return "CRITICAL: Evacuate, isolate ignition, ventilate, escalate response."
    if status == "warning":
        return "Warning: Increase ventilation, investigate source, monitor closely."
    return "Normal: Continue routine monitoring."

def linear_predict(df_room_gas: pd.DataFrame, minutes_ahead=20):
    # Simple linear projection using last 12 points
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
            advice = "Shut down burners, isolate ignition sources, ventilate, evacuate."
        elif room == "Chemical Storage" and g == "H2S" and stt != "safe":
            advice = "Seal room, don PPE (SCBA), isolate ventilation, alert hazmat."
        elif room in ["Pharma Lab A", "Pharma Lab B", "Quality Control Lab"] and stt != "safe":
            advice = "Stop experiments, evacuate, start fume hood purge, investigate."
        elif room == "Cleanroom" and stt != "safe":
            advice = "Evacuate, initiate cleanroom purge, check make-up air, investigate."
        else:
            advice = ("Levels normal. Continue monitoring and verify recent calibration."
                      if stt == "safe" else
                      "Increase ventilation and investigate likely sources; monitor closely.")
        lines.append(f"{g}: {ppm:.0f} ppm — {stt.upper()}. {advice}")
    lines.append("\nPreventative measures:")
    lines.append("• Keep calibration & bump-test schedule up to date.")
    lines.append("• Verify extraction / make-up air performance.")
    lines.append("• Review recent near-misses; update SOPs & training.")
    return "\n".join(lines)

def safety_manual_text(room: str, model: str, gas: str) -> str:
    steps = [
        f"Device: {model} — Location: {room}",
        "1) Confirm reading on secondary display/logs.",
        "2) If WARNING: increase ventilation, start source hunt, brief staff.",
        "3) If CRITICAL: evacuate affected area, isolate ignition sources.",
        f"4) Gas-specific PPE: {'SCBA' if gas=='H2S' else 'Respiratory protection as per CO/CH₄ SOPs'}.",
        "5) Deploy fixed/portable verification (bump check nearby).",
        "6) After control: leak isolation, root-cause analysis, reset and re-arm.",
    ]
    return "\n".join(steps)

def simulate_event(df: pd.DataFrame):
    room = random.choice(PRIMARY_ROOMS)
    gas = random.choice(ROOM_GAS_PROFILE.get(room, GASES))
    now = df["timestamp"].max()
    # Create rising profile over last 6 minutes
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
        "note": "Simulated spike for demo"
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
    # Drop last 8 minutes of that room/gas and rebuild with a spike
    mask = (df["timestamp"]>= now - pd.Timedelta(minutes=7)) & (df["room"]==room) & (df["gas"]==gas)
    df = df[~mask].copy()
    # Rebuild 8 minutes including spike profile
    times = pd.date_range(now - pd.Timedelta(minutes=7), periods=8, freq="1min")
    vals = np.linspace(THRESHOLDS[gas]["warning"]-5, THRESHOLDS[gas]["critical"]+30, num=8)
    for t, v in zip(times, vals):
        df.loc[len(df)] = {"timestamp": t, "room": room, "gas": gas, "ppm": float(v)}
    st.session_state.df = df.sort_values("timestamp")

def build_facility_figure(df: pd.DataFrame):
    fig = go.Figure()

    # Draw rooms (walls)
    for room, (x0, y0, x1, y1) in ROOM_RECTS.items():
        fill = CORRIDOR_FILL if room.startswith("Corridor") else ROOM_FILL
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=NEUTRAL_LINE, width=2),
                      fillcolor=fill)

        # Room label inside (top-left-ish to avoid overlap with dot)
        fig.add_annotation(x=x0 + 2, y=y1 - 3, text=room, showarrow=False,
                           font=dict(size=12, color="#111"), xanchor="left")

    # Stairs (simple symbol)
    fig.add_shape(type="rect", x0=74, y0=2, x1=98, y1=24,
                  line=dict(color=NEUTRAL_LINE, width=2), fillcolor="#f2f2f2")
    fig.add_annotation(x=86, y=13, text="Stairs", showarrow=False, font=dict(size=12))

    # Fire exits (green door blocks)
    exits = [(2, 66, 6, 68), (94, 26, 98, 28)]
    for (x0, y0, x1, y1) in exits:
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

        # Tooltip with prediction info
        hover_lines = [f"<b>{room}</b> — {DETECTOR_MODEL[room]}"]
        for g in GASES:
            ppm, s = vals[g]
            hover_lines.append(f"{g}: {ppm:.0f} ppm ({s})")
        # Add predicted danger time (worst gas)
        worst_eta_txt = ""
        worst_rank = {"safe":0,"warning":1,"critical":2}[stt]
        if worst_rank > 0:
            # check each gas ETA
            eta_texts = []
            for g in GASES:
                dfg = df[(df["room"]==room)&(df["gas"]==g)]
                eta = time_to_next_threshold(dfg, g)
                if eta:
                    level, delta = eta
                    mins = int(max(delta.total_seconds()/60, 0))
                    eta_texts.append(f"{g} → {level.upper()} in ~{mins} min")
            if eta_texts:
                worst_eta_txt = "<br>".join(eta_texts)
        if worst_eta_txt:
            hover_lines.append(f"<i>{worst_eta_txt}</i>")
        # AI short note
        # choose the worst-status gas to comment on
        worst_gas = None
        rank_best = -1
        for g in GASES:
            rank = {"safe":0,"warning":1,"critical":2}[vals[g][1]]
            if rank > rank_best:
                rank_best = rank; worst_gas = g
        ai_note = short_ai_expl(room, worst_gas, vals[worst_gas][0], vals[worst_gas][1]) if worst_gas else "Normal."
        hover_lines.append(ai_note)

        x, y = HOTSPOTS[room]
        xs.append(x); ys.append(y); sizes.append(size); colors.append(color); custom.append(room)
        texts.append("<br>".join(hover_lines))

        # Model label under dot (small text)
        fig.add_annotation(x=x, y=y-4, text=DETECTOR_MODEL[room],
                           showarrow=False, font=dict(size=10, color="#222"),
                           xanchor="center")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="#ffffff")),
        hoverinfo="text", text=texts, customdata=custom
    ))

    fig.update_xaxes(range=[0,100], visible=False)
    fig.update_yaxes(range=[0,100], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=620)

    return fig

def facility_controls_and_legend():
    logo_path = Path("assets/obw_logo.png")
    cols = st.columns([2, 3, 3, 2, 2])
    with cols[0]:
        st.markdown("### OBW — TPPR AI")
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
    with cols[1]:
        st.markdown("**Legend**")
        lg = st.columns(3)
        with lg[0]: st.markdown(f"<div style='color:{SAFE_COLOR}'>● Safe</div>", unsafe_allow_html=True)
        with lg[1]: st.markdown(f"<div style='color:{WARN_COLOR}'>● Warning</div>", unsafe_allow_html=True)
        with lg[2]: st.markdown(f"<div style='color:{CRIT_COLOR}'>● Critical</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("**Actions**")
        if st.button("Simulate Live Gas Event", use_container_width=True):
            simulate_event(st.session_state.df)
        if st.button("Replay Last Incident", use_container_width=True):
            replay_last_incident()
    with cols[3]:
        # Most recent alert preview
        df = st.session_state.df
        alert_txt = "All clear — no current alerts."
        for room in PRIMARY_ROOMS:
            snap = latest_snapshot_for_room(df, room)
            if not snap:
                continue
            _, vals = snap
            stt = overall_status(vals)
            if stt in ("warning", "critical"):
                # choose the worst gas
                worst = max(GASES, key=lambda g: {"safe":0,"warning":1,"critical":2}[vals[g][1]])
                ppm = vals[worst][0]
                alert_txt = f"**Latest Alert:** {room} — {worst} {ppm:.0f} ppm ({stt.upper()})"
                break
        st.info(alert_txt)
    with cols[4]:
        # Incident CSV download
        st.markdown("**Reports**")
        if st.session_state.incidents:
            csv = pd.DataFrame(st.session_state.incidents).to_csv(index=False).encode("utf-8")
            st.download_button("Download Incidents CSV", data=csv, file_name="incidents.csv", mime="text/csv", use_container_width=True)
        else:
            st.caption("No incidents yet.")

# -----------------------------------------------------------------------------
# VIEWS
# -----------------------------------------------------------------------------
def render_facility():
    facility_controls_and_legend()
    fig = build_facility_figure(st.session_state.df)

    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        override_height=640,
        override_width="100%"
    )

    if clicked and len(clicked) > 0 and "customdata" in clicked[0]:
        room_clicked = clicked[0]["customdata"]
        st.session_state.selected_room = room_clicked
        st.session_state.view = "room"

    st.markdown("---")
    st.markdown("#### Incident History")
    if not st.session_state.incidents:
        st.caption("No incidents recorded.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.incidents), use_container_width=True)

def render_room(room: str):
    st.markdown(f"### {room} — {DETECTOR_MODEL.get(room, 'Honeywell Detector')}")
    df = st.session_state.df
    df_room = df[df["room"] == room].copy().sort_values("timestamp")

    c1, c2 = st.columns([3, 2])
    with c1:
        if df_room.empty:
            st.warning("No data available for this room.")
        else:
            fig = px.line(df_room, x="timestamp", y="ppm", color="gas", title="Live Gas Readings")
            # thresholds as lines
            for g in GASES:
                fig.add_hline(y=THRESHOLDS[g]["warning"], line=dict(color=WARN_COLOR, dash="dot"))
                fig.add_hline(y=THRESHOLDS[g]["critical"], line=dict(color=CRIT_COLOR, dash="dot"))
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**AI Assistant — Summary & Recommendations**")
        st.text_area("", ai_room_summary(room, df_room), height=360)

    st.markdown("#### Per-Gas Status & 15-Minute Outlook")
    cols = st.columns(3)
    for i, g in enumerate(GASES):
        with cols[i]:
            dfg = df_room[df_room["gas"] == g]
            if dfg.empty:
                st.metric(g, "—", "No data")
                continue
            last_ppm = float(dfg["ppm"].iloc[-1])
            stt = status_of(g, last_ppm)
            st.metric(g, f"{last_ppm:.0f} ppm", stt.upper())
            pred = linear_predict(dfg, minutes_ahead=15)
            if pred is not None:
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=dfg["timestamp"], y=dfg["ppm"], mode="lines", name="Actual"))
                f2.add_trace(go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name="Projected", line=dict(dash="dash")))
                f2.add_hline(y=THRESHOLDS[g]["warning"], line=dict(color=WARN_COLOR, dash="dot"))
                f2.add_hline(y=THRESHOLDS[g]["critical"], line=dict(color=CRIT_COLOR, dash="dot"))
                f2.update_layout(height=220, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
                st.plotly_chart(f2, use_container_width=True)
                eta = time_to_next_threshold(dfg, g)
                if eta:
                    level, delta = eta
                    mins = int(max(delta.total_seconds()/60, 0))
                    st.warning(f"Projected to reach **{level.upper()}** in ~{mins} min if trend continues.")
                else:
                    st.success("Projection stays below thresholds over the next 15–30 min.")
            else:
                st.caption("Not enough data to project.")

    st.markdown("---")
    c3, c4, c5 = st.columns([1,1,2])
    with c3:
        if st.button("Back to Facility", use_container_width=True):
            st.session_state.view = "facility"
    with c4:
        # Safety manual for the most concerning gas at present
        if not df_room.empty:
            snap_ts = df_room["timestamp"].max()
            snap = df_room[df_room["timestamp"] == snap_ts]
            worst = None; worst_rank = -1
            for g in GASES:
                row = snap[snap["gas"] == g]
                ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
                rank = {"safe":0,"warning":1,"critical":2}[status_of(g, ppm)]
                if rank > worst_rank:
                    worst_rank = rank; worst = g
            if worst:
                manual = safety_manual_text(room, DETECTOR_MODEL.get(room, "Honeywell"), worst)
                st.download_button("View Safety Manual (txt)", manual.encode("utf-8"),
                                   file_name=f"safety_manual_{room.replace(' ','_')}_{worst}.txt",
                                   mime="text/plain", use_container_width=True)
    with c5:
        # Export room series (CSV)
        if not df_room.empty:
            out = df_room.sort_values("timestamp")
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Room Data (CSV)", data=csv,
                               file_name=f"{room.replace(' ','_')}_readings.csv",
                               mime="text/csv", use_container_width=True)

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
if st.session_state.view == "facility":
    render_facility()
else:
    render_room(st.session_state.selected_room or "Boiler Room")


