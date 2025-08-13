
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
from pathlib import Path
import datetime as dt
import random

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="OBW AI Safety Assistant", layout="wide")

ROOMS = [f"Room {i}" for i in range(1, 12)]
GASES = ["Methane", "CO", "H2S"]
THRESHOLDS = {
    "Methane": {"warning": 50, "critical": 100},
    "CO": {"warning": 25, "critical": 50},
    "H2S": {"warning": 10, "critical": 20},
}

# 2D diagram canvas coordinates (0..100 space). Simple rects laid out nicely.
ROOM_RECTS = {
    "Room 1":  (2,  60, 18, 90),
    "Room 2":  (20, 60, 36, 90),
    "Room 3":  (38, 60, 54, 90),
    "Room 4":  (56, 60, 72, 90),
    "Room 5":  (74, 60, 98, 90),
    "Room 6":  (2,  32, 22, 58),
    "Room 7":  (24, 32, 44, 58),
    "Room 8":  (46, 32, 66, 58),
    "Room 9":  (68, 32, 98, 58),
    "Room 10": (2,  4,  42, 30),
    "Room 11": (44, 4,  98, 30),
}

# Detector dot positions (center-ish inside each rect)
HOTSPOTS = {
    "Room 1":  (10, 75),
    "Room 2":  (28, 78),
    "Room 3":  (46, 76),
    "Room 4":  (64, 78),
    "Room 5":  (86, 78),
    "Room 6":  (12, 45),
    "Room 7":  (34, 46),
    "Room 8":  (56, 46),
    "Room 9":  (84, 46),
    "Room 10": (20, 17),
    "Room 11": (70, 17),
}

# ---------------------------
# STATE INIT
# ---------------------------
if "view" not in st.session_state:
    st.session_state.view = "facility"   # "facility" or "room"
if "selected_room" not in st.session_state:
    st.session_state.selected_room = None
if "df" not in st.session_state:
    # Build 60 minutes of sample data (1-min cadence)
    now = pd.Timestamp.now().floor("min")
    ts = pd.date_range(now - pd.Timedelta(minutes=59), periods=60, freq="1min")
    rows = []
    for room in ROOMS:
        for gas in GASES:
            base = 10 if gas == "H2S" else (20 if gas == "CO" else 30)
            noise = np.clip(np.random.normal(0, 5, size=len(ts)), -8, 8)
            vals = np.clip(base + noise, 0, None)
            rows.extend({"timestamp": t, "room": room, "gas": gas, "ppm": float(v)} for t, v in zip(ts, vals))
    st.session_state.df = pd.DataFrame(rows)

if "incidents" not in st.session_state:
    st.session_state.incidents = []  # list of dicts: {time, room, gas, level, ppm, note}

# ---------------------------
# HELPERS
# ---------------------------
def status_of(gas, ppm):
    th = THRESHOLDS[gas]
    if ppm >= th["critical"]:
        return "critical"
    if ppm >= th["warning"]:
        return "warning"
    return "safe"

def color_for_status(status):
    return {"safe": "#26a269", "warning": "#f99b11", "critical": "#cc0000"}[status]

def latest_room_snapshot(df, room):
    r = df[df["room"] == room]
    if r.empty:
        return None
    tmax = r["timestamp"].max()
    snap = r[r["timestamp"] == tmax]
    # dict gas->(ppm,status)
    out = {}
    for g in GASES:
        row = snap[snap["gas"] == g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        out[g] = (ppm, status_of(g, ppm))
    return tmax, out

def short_ai_expl(g, ppm, status):
    if status == "critical":
        return f"{g}: {ppm:.0f} ppm — CRITICAL. Evacuate, isolate ignition, ventilate."
    if status == "warning":
        return f"{g}: {ppm:.0f} ppm — Warning. Increase ventilation, monitor closely."
    return f"{g}: {ppm:.0f} ppm — Normal. Continue monitoring."

def room_hover_text(room, t_and_vals):
    if not t_and_vals:
        return f"{room}<br>No data"
    t, vals = t_and_vals
    parts = [f"<b>{room}</b>", f"<i>{t.strftime('%H:%M')}</i>"]
    for g in GASES:
        ppm, stt = vals[g]
        parts.append(short_ai_expl(g, ppm, stt))
    return "<br>".join(parts)

def overall_status(vals):
    # vals: dict gas -> (ppm,status) ; overall is worst
    if any(s == "critical" for (_, s) in vals.values()):
        return "critical"
    if any(s == "warning" for (_, s) in vals.values()):
        return "warning"
    return "safe"

def simulate_spike():
    df = st.session_state.df
    room = random.choice(ROOMS)
    gas = random.choice(GASES)
    now = df["timestamp"].max()
    # Add spike over last 5 minutes
    for i in range(5):
        t = now - pd.Timedelta(minutes=i)
        idx = (df["timestamp"] == t) & (df["room"] == room) & (df["gas"] == gas)
        if idx.any():
            spike = {"Methane": random.randint(110, 180),
                     "CO": random.randint(55, 120),
                     "H2S": random.randint(22, 60)}[gas]
            df.loc[idx, "ppm"] = spike
    st.session_state.df = df
    # Log incident
    st.session_state.incidents.insert(0, {
        "time": now.strftime("%Y-%m-%d %H:%M"),
        "room": room,
        "gas": gas,
        "level": "Critical",
        "ppm": int(df[(df['timestamp']==now)&(df['room']==room)&(df['gas']==gas)]['ppm'].iloc[0]),
        "note": "Simulated spike for demo"
    })

def ai_room_summary(df_room):
    if df_room.empty:
        return "No data available for this room."
    last_ts = df_room["timestamp"].max()
    snap = df_room[df_room["timestamp"] == last_ts]
    lines = [f"Incident Summary — {df_room['room'].iloc[0]}",
             f"Time: {last_ts.strftime('%Y-%m-%d %H:%M')}"]
    for g in GASES:
        row = snap[snap["gas"] == g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        stt = status_of(g, ppm)
        if stt == "critical":
            advice = ("Evacuate area immediately. Ventilate aggressively. "
                      "Isolate energy/ignition sources. Don PPE (SCBA for H2S). "
                      "Locate and isolate leak.")
        elif stt == "warning":
            advice = ("Increase ventilation and investigate source. "
                      "Prepare to escalate response. Limit non-essential access.")
        else:
            advice = ("Levels normal. Continue routine monitoring. "
                      "Verify detector calibration and ventilation system health.")
        lines.append(f"{g}: {ppm:.0f} ppm — {stt.upper()}. {advice}")
    lines.append("\nGeneral preventative measures:")
    lines.append("- Maintain calibration & bump-test schedule.")
    lines.append("- Verify make-up air and extraction rates.")
    lines.append("- Review recent near-misses and update SOPs.")
    lines.append("- Run refresher training on emergency response.")
    return "\n".join(lines)

def simple_predict_series(df_room_gas, minutes_ahead=15):
    # Very simple linear projection using last 10 points
    if len(df_room_gas) < 3:
        return None
    tail = df_room_gas.sort_values("timestamp").tail(10).copy()
    # convert time to minutes since start
    t0 = tail["timestamp"].min()
    tail["m"] = (tail["timestamp"] - t0).dt.total_seconds() / 60.0
    x = tail["m"].values
    y = tail["ppm"].values
    # linear fit
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # forecast next minutes_ahead
    last_m = x[-1]
    fut_m = np.arange(last_m + 1, last_m + minutes_ahead + 1)
    fut_ts = [tail["timestamp"].iloc[-1] + pd.Timedelta(minutes=int(k - last_m)) for k in fut_m]
    fut_y = m * fut_m + c
    return pd.DataFrame({"timestamp": fut_ts, "ppm": fut_y})

def time_to_threshold(df_room_gas, gas):
    pred = simple_predict_series(df_room_gas, minutes_ahead=30)
    if pred is None or pred.empty:
        return None
    for thresh_name in ["warning", "critical"]:
        th = THRESHOLDS[gas][thresh_name]
        hit = pred[pred["ppm"] >= th]
        if not hit.empty:
            return thresh_name, (hit["timestamp"].iloc[0] - df_room_gas["timestamp"].max())
    return None

# ---------------------------
# FACILITY VIEW (2D DIAGRAM)
# ---------------------------
def render_facility():
    df = st.session_state.df

    top_l, top_r = st.columns([1, 3])
    with top_l:
        st.markdown("### Facility Overview")
        if st.button("Simulate Live Gas Event"):
            simulate_spike()
    with top_r:
        # Most recent alert (if any)
        alert = None
        for room in ROOMS:
            t_and_vals = latest_room_snapshot(df, room)
            if not t_and_vals: 
                continue
            _, vals = t_and_vals
            stt = overall_status(vals)
            if stt in ("warning", "critical"):
                # pick the worst gas
                worst = sorted([(g, *vals[g]) for g in GASES], key=lambda x: ["safe","warning","critical"].index(x[2]))[-1]
                # worst tuple: (gas, ppm, status)
                alert = (room, worst[0], worst[1], worst[2])
                break
        if alert:
            room, gas, ppm, status = alert
            st.info(f"**Most Recent Alert:** {room} — {gas} {ppm:.0f} ppm ({status.upper()})")
        else:
            st.success("All clear — no current alerts.")

    # Build figure
    fig = go.Figure()

    # Draw rooms
    for room, (x0, y0, x1, y1) in ROOM_RECTS.items():
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#888", width=1), fillcolor="#f5f7fa")
        # Label
        fig.add_annotation(x=(x0+x1)/2, y=y1-2, text=room, showarrow=False, font=dict(size=12))

    # Detector dots with color by status
    xs, ys, texts, colors, cust = [], [], [], [], []
    for room in ROOMS:
        t_and_vals = latest_room_snapshot(df, room)
        hover = room_hover_text(room, t_and_vals)
        if t_and_vals:
            _, vals = t_and_vals
            stt = overall_status(vals)
            col = color_for_status(stt)
        else:
            col = "#9aa0a6"
        x, y = HOTSPOTS[room]
        xs.append(x); ys.append(y); texts.append(hover); colors.append(col); cust.append(room)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=16, color=colors, line=dict(width=2, color="#ffffff")),
        hoverinfo="text", text=texts, customdata=cust,
    ))

    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(range=[0, 100], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=550)

    # Capture clicks
    clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=560, override_width="100%")
    if clicked:
        room_clicked = clicked[0]["customdata"]
        st.session_state.selected_room = room_clicked
        st.session_state.view = "room"

    # Incident history (right side)
    st.markdown("---")
    st.markdown("#### Incident History")
    if not st.session_state.incidents:
        st.caption("No incidents recorded yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.incidents)
        st.dataframe(hist_df, use_container_width=True)

# ---------------------------
# ROOM VIEW
# ---------------------------
def render_room(room):
    df = st.session_state.df
    st.markdown(f"### {room}")
    df_room = df[df["room"] == room].copy().sort_values("timestamp")

    c1, c2 = st.columns([3, 2])
    with c1:
        if df_room.empty:
            st.warning("No data available for this room.")
        else:
            fig = px.line(df_room, x="timestamp", y="ppm", color="gas", title="Live Gas Readings")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**AI Assistant — Summary & Recommendations**")
        st.text_area("",
            ai_room_summary(df_room) if not df_room.empty else "No data.",
            height=320
        )

    # Per-gas latest + prediction
    st.markdown("#### Per-Gas Status & 15-minute Outlook")
    cols = st.columns(3)
    for idx, g in enumerate(GASES):
        with cols[idx]:
            dfg = df_room[df_room["gas"] == g]
            if dfg.empty:
                st.metric(g, "—", "No data")
                continue
            last_ppm = float(dfg["ppm"].iloc[-1])
            stt = status_of(g, last_ppm)
            st.metric(g, f"{last_ppm:.0f} ppm", stt.upper())
            # prediction
            pred = simple_predict_series(dfg, minutes_ahead=15)
            if pred is not None:
                figp = go.Figure()
                figp.add_trace(go.Scatter(x=dfg["timestamp"], y=dfg["ppm"], mode="lines", name="Actual"))
                figp.add_trace(go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name="Projected", line=dict(dash="dash")))
                # thresholds
                figp.add_hline(y=THRESHOLDS[g]["warning"], line=dict(color="#f99b11", dash="dot"))
                figp.add_hline(y=THRESHOLDS[g]["critical"], line=dict(color="#cc0000", dash="dot"))
                figp.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(figp, use_container_width=True)
                eta = time_to_threshold(dfg, g)
                if eta:
                    level, delta = eta
                    mins = int(delta.total_seconds()/60)
                    st.warning(f"Likely to reach **{level.upper()}** in ~{mins} min if trend continues.")
                else:
                    st.success("Projection stays below thresholds in next 15–30 min.")
            else:
                st.caption("Not enough data to project.")

    st.markdown("---")
    if st.button("Back to Facility"):
        st.session_state.view = "facility"

# ---------------------------
# ROUTER
# ---------------------------
if st.session_state.view == "facility":
    render_facility()
else:
    render_room(st.session_state.selected_room or "Room 1")

