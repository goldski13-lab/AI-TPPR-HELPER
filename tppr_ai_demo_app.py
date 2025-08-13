
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from PIL import Image
import random
import datetime

# ==============================
# CONFIGURATION
# ==============================
ASSETS_DIR = Path(__file__).parent / "assets"
FACILITY_IMG = ASSETS_DIR / "facility.png"
LOGO_IMG = ASSETS_DIR / "obw_logo.png"
DATA_FILE = Path(__file__).parent / "sample_data.csv"

ROOM_LABELS = {i: f"Room {i}" for i in range(1, 12)}
GASES = ["Methane", "CO", "H2S"]
THRESHOLDS = {
    "Methane": {"warning": 50, "critical": 100},
    "CO": {"warning": 25, "critical": 50},
    "H2S": {"warning": 10, "critical": 20}
}

# Hotspot coordinates (as % of image width/height)
HOTSPOTS = {
    1: (0.15, 0.2),
    2: (0.35, 0.18),
    3: (0.55, 0.22),
    4: (0.75, 0.25),
    5: (0.85, 0.45),
    6: (0.65, 0.55),
    7: (0.45, 0.5),
    8: (0.25, 0.55),
    9: (0.15, 0.75),
    10: (0.35, 0.78),
    11: (0.55, 0.75),
}

# ==============================
# LOAD DATA
# ==============================
if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
else:
    # Create sample data
    timestamps = pd.date_range(datetime.datetime.now() - datetime.timedelta(hours=1), periods=60, freq="T")
    data = []
    for room in range(1, 12):
        for gas in GASES:
            for ts in timestamps:
                data.append({
                    "timestamp": ts,
                    "room": room,
                    "gas": gas,
                    "ppm": random.randint(0, 120)
                })
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)

# ==============================
# UTILS
# ==============================
def status_label(gas, ppm):
    if ppm >= THRESHOLDS[gas]["critical"]:
        return "Critical"
    elif ppm >= THRESHOLDS[gas]["warning"]:
        return "Warning"
    else:
        return "Safe"

def gas_specific_advice(gas, status):
    advice = {
        "Methane": {
            "Critical": "Evacuate immediately, eliminate ignition sources, and ventilate area.",
            "Warning": "Increase ventilation and monitor closely for escalation.",
            "Safe": "Continue monitoring and ensure ventilation systems are functional."
        },
        "CO": {
            "Critical": "Evacuate and ventilate immediately, administer oxygen to affected persons.",
            "Warning": "Investigate source of CO and improve ventilation.",
            "Safe": "Maintain monitoring; check CO sources regularly."
        },
        "H2S": {
            "Critical": "Evacuate, wear SCBA before re-entry, ventilate area.",
            "Warning": "Increase ventilation and check for leaks in process equipment.",
            "Safe": "Continue monitoring and maintain detection systems."
        }
    }
    return advice[gas][status]

def generic_preventative_measures():
    return [
        "Regularly calibrate all gas detectors.",
        "Ensure ventilation systems are operating at full capacity.",
        "Conduct routine safety drills for staff.",
        "Review incident logs and address recurring causes.",
        "Keep emergency PPE accessible at all times."
    ]

def build_incident_summary(room_id, when):
    names = ROOM_LABELS
    room_df = df[(df["room"] == room_id) & (df["timestamp"] <= when)].sort_values("timestamp")
    if room_df.empty or room_df["timestamp"].isna().all():
        return f"Incident — {names[room_id]}\nNo data available for this location."
    
    last_ts = room_df["timestamp"].max()
    snapshot = room_df[room_df["timestamp"] == last_ts]
    day_start = last_ts.floor("D")
    lines = [f"Incident — {names[room_id]}", f"Time: {last_ts.strftime('%Y-%m-%d %H:%M')}"]

    for g in GASES:
        gday = df[(df['room'] == room_id) & (df['gas'] == g) & (df['timestamp'] >= day_start) & (df['timestamp'] <= last_ts)]
        w = int((gday["ppm"] >= THRESHOLDS[g]["warning"]).sum())
        c = int((gday["ppm"] >= THRESHOLDS[g]["critical"]).sum())
        row = snapshot[snapshot["gas"] == g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        label = status_label(g, ppm).title()
        extra = f" — {c} critical event(s) today" if label == "Critical" else (f" — {w} warning event(s) today" if label == "Warning" else "")
        lines.append(f"{g}: {ppm} ppm ({label}){extra}")
        lines.append(f"   Advice: {gas_specific_advice(g, label)}")
    lines.append("\nGeneral Preventative Measures:")
    for m in generic_preventative_measures():
        lines.append(f"- {m}")
    return "\n".join(lines)

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="OBW AI Safety Assistant", layout="wide")

# Load images
facility_img = Image.open(FACILITY_IMG)
logo_img = Image.open(LOGO_IMG)

# Sidebar logo
st.sidebar.image(logo_img, use_container_width=True)
st.sidebar.title("OBW AI Safety Assistant")

# State
if "view" not in st.session_state:
    st.session_state.view = "facility"
if "selected_room" not in st.session_state:
    st.session_state.selected_room = None

# Facility View
if st.session_state.view == "facility":
    st.image(facility_img, use_container_width=True)
    st.markdown("### Click on a room number to view details")
    cols = st.columns(6)
    for i in range(1, 12):
        with cols[(i-1) % 6]:
            if st.button(f"Room {i}"):
                st.session_state.selected_room = i
                st.session_state.view = "room"

# Room View
if st.session_state.view == "room":
    room_id = st.session_state.selected_room
    st.markdown(f"## {ROOM_LABELS[room_id]}")
    room_df = df[df["room"] == room_id]
    if room_df.empty:
        st.warning("No data available for this room.")
    else:
        fig = px.line(room_df, x="timestamp", y="ppm", color="gas", title=f"Gas Readings — {ROOM_LABELS[room_id]}")
        st.plotly_chart(fig, use_container_width=True)
        summary = build_incident_summary(room_id, room_df["timestamp"].max())
        st.text_area("AI Safety Summary", summary, height=300)
    if st.button("Back to Facility"):
        st.session_state.view = "facility"

