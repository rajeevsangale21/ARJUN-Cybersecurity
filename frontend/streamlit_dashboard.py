from __future__ import annotations

import base64
import json
import pickle

try:
    import joblib
except Exception:
    joblib = None
from datetime import datetime
from pathlib import Path
import sys

# Ensure sibling ARJUN packages are importable when Streamlit Cloud
# executes this file from the frontend directory.
FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from alerts.alert_engine import AlertEngine
except Exception:
    AlertEngine = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# PATHS
# ============================================================
FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "saved_models"
ASSETS = ROOT / "frontend" / "assets"

PRODUCTION_JSON = DATA / "production_forecast.json"
PRODUCTION_NPZ = DATA / "production_forecast.npz"
EXPLAIN_JSON = DATA / "explainability_report.json"
ZEEK_STATES = DATA / "zeek_network_states.npz"
ZEEK_GRAPHS = DATA / "zeek_network_graphs.pkl"
ZEEK_FORECAST_NPZ = DATA / "zeek_world_model_forecast.npz"
ZEEK_RISK_JSON = DATA / "zeek_forecast_risk.json"
ZEEK_RISK_NPZ = DATA / "zeek_forecast_risk.npz"
ALERT_HISTORY = ROOT / "alerts" / "zeek_alert_history.json"
LIVE_STATUS = DATA / "zeek_live_status.json"
WORLD_MODEL = MODELS / "hybrid_world_model.pt"
XGB_MODEL = MODELS / "xgboost_attack_classifier_improved.pkl"

ICON_CANDIDATES = [
    ASSETS / "arjun_icon.png",
    FILE_DIR / "assets" / "arjun_icon.png",
    ROOT / "assets" / "arjun_icon.png",
    ROOT / "arjun_icon.png",
    Path("/mnt/data/994d8db2-37e6-474b-83a9-75aa50a122a1.png"),
]


# ============================================================
# PAGE / THEME
# ============================================================
st.set_page_config(
    page_title="ARJUN | Predictive Cyber Defence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    r"""
<style>
:root{
  --navy:#03152F;
  --navy2:#061F43;
  --panel:#062A55;
  --panel2:#052244;
  --blue:#087FF5;
  --cyan:#19D8FF;
  --cyan2:#62E9FF;
  --text:#F3FAFF;
  --muted:#9FC0D6;
  --line:rgba(88,204,255,.25);
  --green:#48E6A0;
  --amber:#FFBF45;
  --red:#FF4F6D;
}
*{box-sizing:border-box;font-family:Arial,Helvetica,sans-serif!important}
html,body,.stApp{font-family:Arial,Helvetica,sans-serif!important}
.stApp{
  background:
    radial-gradient(850px 520px at 82% 4%,rgba(0,128,255,.32),transparent 64%),
    radial-gradient(650px 460px at 4% 44%,rgba(0,215,255,.12),transparent 66%),
    linear-gradient(135deg,#03152F 0%,#073A76 48%,#05285A 100%);
  color:var(--text);
}
.stApp:before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(97,211,255,.045) 1px,transparent 1px),
    linear-gradient(90deg,rgba(97,211,255,.045) 1px,transparent 1px);
  background-size:36px 36px;
  mask-image:linear-gradient(to bottom,black,transparent 90%);
}
.block-container{max-width:1800px;padding:14px 22px 42px;position:relative;z-index:1}
header[data-testid="stHeader"]{background:transparent}
section[data-testid="stSidebar"]{display:none!important}
#MainMenu,footer{visibility:hidden}

/* ---------------- HERO ---------------- */
.brand-bar{position:relative;overflow:visible;min-height:440px;padding:34px 42px;margin-bottom:24px;border:1px solid rgba(78,207,255,.38);border-radius:24px;background:linear-gradient(105deg,rgba(2,24,55,.98),rgba(4,62,123,.94) 55%,rgba(4,29,67,.96));box-shadow:0 25px 70px rgba(0,8,30,.36),inset 0 1px 0 rgba(255,255,255,.07)}
.brand-bar:before{
  content:"";position:absolute;left:-5%;bottom:-120px;width:72%;height:220px;
  border-top:4px solid rgba(24,216,255,.45);border-radius:50%;transform:rotate(-5deg);
  box-shadow:0 -20px 0 rgba(30,161,255,.14),0 -42px 0 rgba(30,161,255,.08),0 0 45px rgba(24,216,255,.16);
}
.brand-bar:after{
  content:"";position:absolute;right:-60px;top:-155px;width:470px;height:470px;
  border:2px solid rgba(78,207,255,.16);border-radius:50%;
  box-shadow:0 0 0 28px rgba(78,207,255,.035),0 0 0 58px rgba(78,207,255,.02),0 0 60px rgba(30,170,255,.12);
}
.brand-left{position:relative;z-index:2;display:flex;gap:18px;align-items:flex-start;max-width:61%;}
.brand-logo{
  width:118px;height:118px;flex:0 0 118px;object-fit:cover;border-radius:18px;
  filter:drop-shadow(0 0 22px rgba(25,216,255,.22));
  mix-blend-mode:screen;
}
.brand-logo-fallback{
  width:118px;height:118px;flex:0 0 118px;border-radius:18px;
  display:flex;align-items:center;justify-content:center;
  background:radial-gradient(circle,#0B83C5 0%,#063566 68%,#041B39 100%);
  border:1px solid rgba(100,220,255,.28);color:white;font-size:25px;font-weight:900;
}
.brand-kicker{
  display:inline-block;padding:9px 14px;border-radius:8px;
  background:linear-gradient(90deg,#0BCBFA,#087BEF);color:white;font-size:15px;font-weight:900;letter-spacing:.07em;
}
.brand-name{color:white;font-size:46px;line-height:1.02;font-weight:900;letter-spacing:-.025em;margin-top:14px}
.brand-sub{color:#C8E8F8;font-size:21px;font-weight:700;margin-top:14px}
.brand-tagline{color:#F4FBFF;font-size:17px;line-height:1.55;max-width:900px;margin-top:12px}

.hero-flow{
  position:absolute;right:28px;top:28px;width:425px;z-index:4;
  display:grid;gap:12px;
}
.hero-node{
  display:flex;align-items:center;gap:12px;padding:10px 13px;
  min-height:62px;border:1px solid rgba(103,216,255,.27);
  background:linear-gradient(110deg,rgba(3,28,62,.91),rgba(5,48,94,.78));
  backdrop-filter:blur(10px);border-radius:15px;color:#E9FAFF;
  box-shadow:0 10px 24px rgba(0,10,30,.14),inset 0 1px 0 rgba(255,255,255,.04);
}
.hero-node-icon{
  width:40px;height:40px;flex:0 0 40px;border-radius:12px;
  background:linear-gradient(145deg,#087FF5,#12D8FF);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;font-weight:900;color:white;box-shadow:0 0 20px rgba(24,216,255,.18)
}
.hero-node b{display:block;font-size:14px;color:#F4FBFF}.hero-node span{display:block;color:#91BBD3;font-size:11px;margin-top:3px}

/* STATUS IS OUTSIDE HERO - NEVER OVERLAPS TAKE ACTION */
.external-status{display:flex;justify-content:center;align-items:center;gap:9px;margin:0;width:100%;padding:9px 13px;position:relative;z-index:6;border:1px solid rgba(117,220,255,.30);border-radius:999px;background:linear-gradient(105deg,rgba(3,24,53,.98),rgba(4,42,78,.96));color:#D7F6FF;font-size:11px;font-weight:900;letter-spacing:.01em;box-shadow:0 8px 24px rgba(0,8,25,.18)}
.hero-flow .external-status{min-height:38px}
.status-dot{width:9px;height:9px;border-radius:50%;display:inline-block;background:var(--green);box-shadow:0 0 0 4px rgba(72,230,160,.10),0 0 18px rgba(72,230,160,.75)}

/* ---------------- TOP NAV ---------------- */
.stTabs [data-baseweb="tab-list"]{
  gap:4px;padding:5px;background:rgba(2,25,57,.95);
  border:1px solid rgba(85,205,255,.30);border-radius:14px;
  box-shadow:0 12px 32px rgba(0,10,30,.25);
}
.stTabs [data-baseweb="tab"]{
  height:50px;padding:0 20px;border-radius:10px;color:#F2FBFF!important;
  -webkit-text-fill-color:#F2FBFF!important;opacity:1!important;
  font-size:15px!important;font-weight:800!important;
}
.stTabs [data-baseweb="tab"] *{color:#F2FBFF!important;-webkit-text-fill-color:#F2FBFF!important;opacity:1!important;font-weight:800!important}
.stTabs [data-baseweb="tab"]:hover{background:rgba(20,215,255,.11)!important}
.stTabs [aria-selected="true"]{background:linear-gradient(105deg,#087DEB,#18D8FF)!important;color:white!important;box-shadow:0 7px 24px rgba(18,211,255,.24)}
.stTabs [aria-selected="true"] *{color:white!important;-webkit-text-fill-color:white!important}

/* ---------------- FILE INGESTION / UPLOADER ---------------- */
/* Keep Streamlit's native uploader inside ARJUN's dark cyber theme. */
[data-testid="stFileUploader"]{width:100%!important}
[data-testid="stFileUploader"] label{color:#DDF5FF!important;font-family:Arial,Helvetica,sans-serif!important;font-weight:900!important;font-size:13px!important;margin-bottom:7px!important}
[data-testid="stFileUploader"] section{
  background:linear-gradient(145deg,rgba(4,39,78,.98),rgba(3,25,53,.98))!important;
  border:1px dashed rgba(25,216,255,.55)!important;
  border-radius:14px!important;
  padding:18px!important;
  min-height:112px!important;
  box-shadow:inset 0 0 28px rgba(8,127,245,.07)!important;
}
[data-testid="stFileUploader"] section > div{background:transparent!important}
[data-testid="stFileUploader"] section div{color:#DDF5FF!important}
[data-testid="stFileUploader"] small{color:#8FB5CB!important}
/* Use Streamlit's native uploader button exactly once. Do not create a pseudo-button. */
/* Native Streamlit uploader: exactly ONE browse control.
   Do not style/hide arbitrary uploader descendants because Streamlit versions
   can render the browse label in more than one nested element. */
[data-testid="stFileUploader"] button{
  position:relative!important;
  background:linear-gradient(105deg,#087FF5,#19D8FF)!important;
  color:transparent!important;
  -webkit-text-fill-color:transparent!important;
  border:1px solid rgba(130,235,255,.65)!important;
  border-radius:9px!important;
  min-height:42px!important;
  min-width:150px!important;
  padding:0 16px!important;
  font-size:0!important;
  font-family:Arial,Helvetica,sans-serif!important;
  font-weight:900!important;
  box-shadow:0 7px 20px rgba(8,127,245,.20)!important;
  overflow:hidden!important;
}
[data-testid="stFileUploader"] button *{
  color:transparent!important;
  -webkit-text-fill-color:transparent!important;
  font-size:0!important;
}
[data-testid="stFileUploader"] button::after{
  content:"CHOOSE FILE";
  position:absolute!important;
  inset:0!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  color:#FFFFFF!important;
  -webkit-text-fill-color:#FFFFFF!important;
  font:900 13px Arial,Helvetica,sans-serif!important;
  letter-spacing:.02em!important;
  pointer-events:none!important;
}
[data-testid="stFileUploader"] button:hover{filter:brightness(1.08)!important}

/* The help icon is intentionally not used by ARJUN's uploader. */
[data-testid="stFileUploader"] [data-testid="stTooltipIcon"]{display:none!important}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]{color:#B8D7E8!important}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#B8D7E8!important;-webkit-text-fill-color:#B8D7E8!important}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] svg{fill:#19D8FF!important;color:#19D8FF!important}
.ingest-shell{
  background:linear-gradient(145deg,rgba(5,43,85,.96),rgba(3,27,58,.98));
  border:1px solid rgba(91,203,255,.27);border-radius:16px;
  padding:18px 20px 16px;margin:4px 0 20px;
  box-shadow:0 12px 30px rgba(0,10,30,.16),inset 0 1px 0 rgba(255,255,255,.035)
}
.ingest-head{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:12px}
.ingest-title{color:#F1FBFF;font-size:17px;font-weight:900;letter-spacing:.03em}
.ingest-sub{color:#91B4C9;font-size:12px;line-height:1.5;margin-top:4px}
.ingest-info{color:#A9D5E9;font-size:11px;text-align:right;line-height:1.5;white-space:nowrap}
.ingest-info b{color:#19D8FF}
.ingest-formats{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
.ingest-chip{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(7,74,126,.42);border:1px solid rgba(75,205,255,.24);color:#BDE8F7;font-size:10px;font-weight:800}
.ingest-action{margin-top:11px}
.ingest-result{margin-top:12px}
@media(max-width:900px){.ingest-head{display:block}.ingest-info{text-align:left;margin-top:7px}.hero-flow{position:relative;right:auto;top:auto;width:100%;margin-top:24px}.brand-left{max-width:100%}.brand-bar{min-height:0}.system-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}

/* ---------------- HARD DARK MODE OVERRIDES ---------------- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"], .stDataFrame, .stDataEditor,
[data-testid="stTable"], [data-testid="stTable"] *{
  background:#061F43!important;color:#EAF7FF!important;
}
[data-testid="stPlotlyChart"]{
  background:linear-gradient(145deg,rgba(3,28,61,.98),rgba(2,20,43,.98))!important;
  border:1px solid rgba(91,203,255,.23)!important;border-radius:16px!important;
  padding:10px!important;overflow:hidden!important;
}
[data-testid="stPlotlyChart"] iframe{background:transparent!important}
.dark-table-wrap table, .dark-table-wrap th, .dark-table-wrap td{
  color:#EAF7FF!important;
}
/* ---------------- TEXT ---------------- */
.stMarkdown,.stMarkdown p,.stMarkdown li,[data-testid="stMarkdownContainer"]{color:#EAF7FF}
h3{color:#F3FAFF!important;font-weight:900!important}
.stCaption,[data-testid="stCaptionContainer"]{color:#9FC0D6!important}
.section-title{color:#E9FAFF;font-size:15px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;margin:23px 0 11px}
.section-title:before{content:"◆";color:var(--cyan);margin-right:9px;font-size:9px}

/* ---------------- STATUS / VALUE CARDS ---------------- */
.system-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:0 0 18px}
.system-pill{background:linear-gradient(145deg,rgba(5,43,85,.86),rgba(3,27,58,.92));border:1px solid rgba(91,203,255,.23);border-radius:12px;padding:12px 14px}
.system-pill .s-label{font-size:9px;color:#82A9C1;text-transform:uppercase;letter-spacing:.08em;font-weight:900}
.system-pill .s-value{font-size:14px;color:#F1FBFF;font-weight:900;margin-top:4px}.system-pill .online{color:#48E6A0}.system-pill .cyan{color:#19D8FF}
.value-card{background:linear-gradient(145deg,rgba(8,54,108,.94),rgba(3,28,61,.97));border:1px solid rgba(98,211,255,.27);border-radius:15px;padding:18px;min-height:128px;box-shadow:0 13px 34px rgba(0,10,30,.18),inset 0 1px 0 rgba(255,255,255,.04)}
.value-card .vlabel{color:#B8D7E8;font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:900}.value-card .vvalue{color:#fff;font-size:31px;font-weight:900;margin-top:10px}.value-card .vmeta{color:#8EADC2;font-size:11px;margin-top:8px}

/* ---------------- PANELS / ALERTS ---------------- */
.panel{background:linear-gradient(145deg,rgba(7,45,90,.91),rgba(3,27,58,.96));border:1px solid rgba(91,203,255,.25);border-radius:15px;padding:18px;box-shadow:0 12px 30px rgba(0,10,30,.17);margin-top:16px}
.panel-title{color:#EAF9FF;font-size:15px;font-weight:900}.panel-meta{color:#91B4C9;font-size:12px;margin-top:4px;line-height:1.55}
.safe,.warn,.danger{border-radius:12px;padding:12px 16px;margin:14px 0;border:1px solid}.safe{color:#7CFFBF;background:rgba(5,74,56,.62);border-color:rgba(72,230,160,.35)}.warn{color:#FFE29B;background:rgba(75,57,12,.60);border-color:rgba(255,191,69,.36)}.danger{color:#FFB8C4;background:rgba(78,17,37,.64);border-color:rgba(255,79,109,.40)}

/* ---------------- DARK TABLES ---------------- */
.dark-table-wrap{overflow-x:auto;border:1px solid rgba(91,203,255,.22);border-radius:13px;margin-top:12px;background:rgba(2,20,43,.72)}
.dark-table{width:100%;border-collapse:collapse;background:transparent!important;font-size:12px}
.dark-table th{padding:12px 11px;text-align:left;color:#9FC5DA;background:linear-gradient(180deg,rgba(8,62,116,.96),rgba(5,44,85,.96));font-size:10px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid rgba(83,198,255,.28)}
.dark-table td{padding:11px;border-top:1px solid rgba(91,203,255,.11);color:#E8F6FC;background:rgba(3,25,52,.72)}
.dark-table tr:nth-child(even) td{background:rgba(5,38,75,.70)}
.dark-table td.ready{color:#48E6A0;font-weight:900}.dark-table td.missing{color:#FF8296;font-weight:900}

/* ---------------- FORECAST ---------------- */
.forecast-panel{background:linear-gradient(145deg,rgba(4,37,75,.95),rgba(2,22,48,.97));border:1px solid rgba(91,203,255,.23);border-radius:16px;padding:18px;margin-top:12px}
.forecast-table{width:100%;border-collapse:collapse;font-size:13px}.forecast-table th{color:#8FB3C8;text-transform:uppercase;letter-spacing:.05em;font-size:10px;text-align:left;padding:10px;border-bottom:1px solid var(--line)}.forecast-table td{color:#E5F4FB;padding:11px 10px;border-bottom:1px solid rgba(105,180,225,.12)}.forecast-step{color:var(--cyan);font-weight:900}
.stage-track{display:flex;gap:10px;overflow-x:auto;padding:4px 0 8px}.stage-chip{min-width:150px;padding:14px;border:1px solid rgba(103,205,255,.23);border-radius:12px;background:rgba(6,40,79,.80)}.stage-chip span{display:block;color:#80A8BF;font-size:10px;font-weight:900}.stage-chip b{display:block;color:#E1F4FC;font-size:13px;margin-top:6px}.stage-chip.active{border-color:#25D9FF;background:linear-gradient(135deg,rgba(12,120,194,.45),rgba(20,57,122,.70));box-shadow:0 0 25px rgba(24,217,255,.14)}

/* ---------------- ALERTS ---------------- */
.alert-center{background:linear-gradient(145deg,rgba(5,36,73,.97),rgba(2,22,48,.99));border:1px solid rgba(94,205,255,.25);border-radius:16px;padding:17px;margin-top:12px;box-shadow:0 14px 35px rgba(0,10,30,.20)}
.alert-row{display:grid;grid-template-columns:1.35fr .55fr .65fr .65fr 1.25fr 1.55fr;gap:12px;align-items:center;padding:10px 7px}
.alert-id{color:#EAF8FF;font-weight:900;font-size:13px}.alert-time{color:#86AFC6;font-size:10px;margin-top:4px}.alert-action{color:#B8D5E5;font-size:11px;line-height:1.45}.alert-action b{color:#E7F7FD}
.badge{display:inline-block;padding:5px 9px;border-radius:999px;font-size:10px;font-weight:900;letter-spacing:.05em}.badge-critical,.badge-high{color:#FFD7DE;background:rgba(255,79,109,.15);border:1px solid rgba(255,79,109,.40)}.badge-medium{color:#FFE29B;background:rgba(255,191,69,.14);border:1px solid rgba(255,191,69,.35)}.badge-low{color:#8DFFD0;background:rgba(72,230,160,.12);border:1px solid rgba(72,230,160,.30)}.badge-predicted{color:#BDEFFF;background:rgba(25,216,255,.10);border:1px solid rgba(25,216,255,.30)}
.alert-empty{padding:28px;text-align:center;color:#9AB9CB;border:1px dashed rgba(94,205,255,.22);border-radius:12px;background:rgba(3,22,48,.42)}
.recommendation{margin-top:15px;padding:13px 15px;background:rgba(3,23,50,.72);border:1px solid rgba(103,205,255,.20);border-radius:9px}.recommendation-title{color:#9EC1D7;font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:900}.recommendation p{color:#C4DAE8;font-size:12px;line-height:1.55;margin:7px 0 0}

/* ---------------- STREAMLIT CONTROLS ---------------- */
.stButton>button{font-family:Arial,Helvetica,sans-serif!important;border-radius:10px!important;min-height:40px!important;background:linear-gradient(100deg,#0877DF,#10BEEB)!important;border:1px solid rgba(111,224,255,.36)!important;color:white!important;font-weight:900!important;box-shadow:0 7px 18px rgba(0,20,55,.20)!important}
.stButton>button:hover{filter:brightness(1.08)!important;transform:translateY(-1px)}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(8,54,108,.94),rgba(3,28,61,.97))!important;border:1px solid rgba(98,211,255,.27)!important;border-radius:15px!important;padding:17px!important;min-height:108px!important}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#BFE0EF!important}[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#FFF!important;-webkit-text-fill-color:#FFF!important;font-weight:900!important}

/* Plotly's iframe/container background is transparent; this also prevents white Streamlit blocks */
[data-testid="stPlotlyChart"]{background:transparent!important;border:0!important}

@media(max-width:1200px){
  .brand-left{max-width:55%}.brand-name{font-size:38px}.hero-flow{width:370px}.brand-logo,.brand-logo-fallback{width:96px;height:96px;flex-basis:96px}
  .system-strip{grid-template-columns:repeat(3,1fr)}
  .alert-row{grid-template-columns:1fr 1fr 1fr}.alert-action{grid-column:auto}
}
@media(max-width:850px){
  .brand-bar{min-height:650px;padding:25px}.brand-left{max-width:100%}.hero-flow{position:absolute;left:25px;right:25px;top:285px;width:auto}.brand-name{font-size:32px}.brand-tagline{font-size:14px}.external-status{margin-right:0}.system-strip{grid-template-columns:1fr 1fr}
}
@media(max-width:600px){.block-container{padding:8px}.brand-bar{min-height:700px}.brand-logo,.brand-logo-fallback{width:78px;height:78px;flex-basis:78px}.brand-kicker{font-size:11px}.brand-name{font-size:27px}.brand-sub{font-size:16px}.hero-flow{top:275px}.system-strip{grid-template-columns:1fr}.external-status{font-size:10px}.alert-row{grid-template-columns:1fr 1fr}
}

.live-alert-card{margin:12px 0 18px;padding:17px 19px;border:1px solid rgba(255,191,69,.30);border-radius:16px;background:linear-gradient(105deg,rgba(49,35,11,.52),rgba(4,34,67,.92));box-shadow:0 12px 32px rgba(0,6,20,.22),inset 0 1px 0 rgba(255,255,255,.035)}
.live-alert-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px;color:#E8F8FF;font-size:14px;letter-spacing:.04em}
.live-alert-top b{margin-right:auto}
.live-alert-status{color:#FFCF78;font-size:11px;font-weight:900;letter-spacing:.08em}
.live-alert-grid{display:grid;grid-template-columns:1.5fr 1fr 1fr 1.5fr;gap:12px}
.live-alert-grid div{padding:10px 12px;border-radius:10px;background:rgba(2,19,43,.66);border:1px solid rgba(104,190,228,.10);min-width:0}
.live-alert-grid small{display:block;color:#7097B4;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.live-alert-grid strong{display:block;color:#DDF5FF;font-size:13px;overflow-wrap:anywhere}
@media(max-width:900px){.live-alert-grid{grid-template-columns:1fr 1fr}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def finite(x):
    try:
        return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.asarray([], dtype=np.float64)


def npz_scalar(container, key, default=0.0):
    try:
        if container is None or key not in container:
            return default
        value = np.asarray(container[key])
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_npz(path):
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return {}


def load_graph_count(path):
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ("graphs", "graph_sequence", "sequence"):
                if key in obj:
                    obj = obj[key]
                    break
        return len(obj)
    except Exception:
        return 0


def live_is_online(status):
    if not isinstance(status, dict):
        return False
    state = str(status.get("status", status.get("state", ""))).upper()
    return state not in {"ERROR", "STOPPED", "OFFLINE"}


def risk_class(score):
    score = float(score)
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    if score >= 0.20:
        return "GUARDED"
    return "LOW"


def normalize_probs(value):
    arr = finite(value).reshape(-1)
    # Production artifacts contain fractions (0.16 = 16%). If someone stores
    # percentages, normalize them safely for display and risk logic.
    if len(arr) and np.nanmax(np.abs(arr)) > 1.5:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def normalize_alerts(value):
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("alerts") or value.get("history") or value.get("items") or []
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def alert_severity(alert):
    return str(alert.get("severity", alert.get("risk_level", "UNKNOWN"))).upper()


def alert_badge_class(severity):
    return {"CRITICAL":"badge-critical","HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}.get(severity, "badge-predicted")


def alert_time(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone().strftime("%d %b %Y · %H:%M:%S")
    except Exception:
        return str(value)[:24] if value else "Time unavailable"


def save_alert_history(items):
    ALERT_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    tmp = ALERT_HISTORY.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    tmp.replace(ALERT_HISTORY)


def render_dark_table(df):
    if df is None or len(df) == 0:
        st.markdown('<div class="alert-empty">No records available.</div>', unsafe_allow_html=True)
        return
    frame = df.copy()
    html = frame.to_html(index=False, escape=True, classes="dark-table")
    html = html.replace(">READY<", ' class="ready">READY<')
    html = html.replace(">ONLINE<", ' class="ready">ONLINE<')
    html = html.replace(">ACTIVE<", ' class="ready">ACTIVE<')
    html = html.replace(">MISSING<", ' class="missing">MISSING<')
    st.markdown(f'<div class="dark-table-wrap">{html}</div>', unsafe_allow_html=True)


def render_dark_forecast_table(timeline, future_probs):
    if not timeline:
        st.markdown('<div class="alert-empty">Forecast timeline is not available.</div>', unsafe_allow_html=True)
        return
    rows = []
    for i, item in enumerate(timeline):
        p = safe_float(item.get("attack_probability", future_probs[i] if i < len(future_probs) else 0.0))
        r = safe_float(item.get("risk_score", 0.0))
        rows.append({
            "Step": f"t+{item.get('step', i+1)}",
            "Attack": f"{p:.2%}",
            "Risk": f"{r:.2%}",
            "Level": str(item.get("risk_level", risk_class(r))).upper(),
            "MITRE": str(item.get("mitre_stage", "Unknown")),
        })
    render_dark_table(pd.DataFrame(rows))


def render_plotly_forecast(future_probs, risk_scores, title="Future Threat Trajectory", height=390):
    if not len(future_probs):
        st.markdown('<div class="alert-empty">No forecast values are available.</div>', unsafe_allow_html=True)
        return
    if go is None:
        st.info("Plotly is not available; forecast values are shown below.")
        return
    n = len(future_probs)
    risks = np.asarray(risk_scores[:n] if len(risk_scores) else np.zeros(n), dtype=float)
    x = [f"t+{i}" for i in range(1, n + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=future_probs*100, mode="lines+markers", name="Attack probability",
                             line=dict(color="#19D8FF", width=4), marker=dict(size=9, color="#19D8FF", line=dict(color="#03152F", width=3))))
    fig.add_trace(go.Scatter(x=x, y=risks*100, mode="lines+markers", name="Risk score",
                             line=dict(color="#087FF5", width=4), marker=dict(size=9, color="#087FF5", line=dict(color="#03152F", width=3))))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#EAF7FF", size=15)),
        height=height,
        margin=dict(l=30,r=20,t=45,b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,25,53,.62)",
        font=dict(family="Arial", color="#DDEFFA"),
        legend=dict(orientation="h", y=1.08, x=0.0, font=dict(color="#DDEFFA")),
        xaxis=dict(color="#9FC0D6", gridcolor="rgba(105,180,225,.12)", linecolor="rgba(105,180,225,.20)", fixedrange=True),
        yaxis=dict(title="Percent", color="#9FC0D6", gridcolor="rgba(105,180,225,.12)", zeroline=False, fixedrange=True),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})


def icon_html():
    for path in ICON_CANDIDATES:
        if path.exists():
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                return f'<img class="brand-logo" src="data:image/png;base64,{b64}">'
            except Exception:
                pass
    return '<div class="brand-logo-fallback">AJ</div>'


# ============================================================
# LOAD ALL ARTIFACTS ONCE
# ============================================================
production = load_json(PRODUCTION_JSON)
prod_npz = load_npz(PRODUCTION_NPZ)
explain = load_json(EXPLAIN_JSON)
risk_json = load_json(ZEEK_RISK_JSON)
status = load_json(LIVE_STATUS)
alerts = normalize_alerts(load_json(ALERT_HISTORY))
states_npz = load_npz(ZEEK_STATES)

# ----- current classifier score -----
classifier = production.get("classifier", {}) if isinstance(production, dict) else {}
current_prob = safe_float(classifier.get("current_attack_probability", npz_scalar(prod_npz, "current_attack_probability", 0.0)))
current_prob = float(np.clip(current_prob, 0.0, 1.0))
threshold = safe_float(classifier.get("threshold", 0.55), 0.55)
current_family = str(classifier.get("current_attack_family", "Unknown"))

# ----- future probabilities: production -> NPZ -> Zeek risk JSON -> Zeek forecast NPZ -----
future_probs = normalize_probs(production.get("future_attack_probabilities", []))
if not len(future_probs):
    future_probs = normalize_probs(prod_npz.get("future_attack_probability", []))
if not len(future_probs):
    future_probs = normalize_probs(prod_npz.get("future_attack_probabilities", []))
if not len(future_probs) and isinstance(risk_json, dict):
    future_probs = normalize_probs(risk_json.get("attack_probability", risk_json.get("probabilities", [])))
if not len(future_probs):
    zf = load_npz(ZEEK_FORECAST_NPZ)
    for key in ("attack_probability", "attack_probabilities", "future_attack_probability", "future_attack_probabilities"):
        if key in zf:
            future_probs = normalize_probs(zf[key])
            if len(future_probs):
                break

# ----- timeline: production -> risk JSON -----
timeline = production.get("timeline", []) if isinstance(production, dict) else []
if not isinstance(timeline, list):
    timeline = []
if not timeline and isinstance(risk_json, dict):
    timeline = risk_json.get("timeline", risk_json.get("risk_timeline", [])) or []
if not isinstance(timeline, list):
    timeline = []

# ----- if timeline is absent, build it from risk arrays -----
if not timeline and len(future_probs):
    rs = risk_json.get("risk_score", risk_json.get("risk_scores", [])) if isinstance(risk_json, dict) else []
    stages = risk_json.get("mitre_stage", risk_json.get("stages", [])) if isinstance(risk_json, dict) else []
    levels = risk_json.get("risk_level", risk_json.get("risk_levels", [])) if isinstance(risk_json, dict) else []
    rs = finite(rs).reshape(-1)
    timeline = []
    for i, p in enumerate(future_probs):
        score = safe_float(rs[i], 0.0) if i < len(rs) else 0.0
        stage = str(stages[i]) if i < len(stages) else "Unknown"
        level = str(levels[i]).upper() if i < len(levels) else risk_class(score)
        timeline.append({"step":i+1,"attack_probability":float(p),"risk_score":score,"risk_level":level,"mitre_stage":stage})

risk_scores = np.asarray([safe_float(x.get("risk_score", 0.0)) for x in timeline], dtype=float) if timeline else np.zeros(len(future_probs))
risk_scores = np.clip(risk_scores, 0.0, 1.0)
if len(future_probs) and len(risk_scores) < len(future_probs):
    risk_scores = np.pad(risk_scores, (0, len(future_probs)-len(risk_scores)), constant_values=0.0)

if current_prob == 0.0 and len(future_probs):
    current_prob = float(future_probs[0])
peak_future = float(np.max(future_probs)) if len(future_probs) else 0.0
peak_risk = float(np.max(risk_scores)) if len(risk_scores) else 0.0
latest_stage = str(timeline[-1].get("mitre_stage", "Unknown")) if timeline else "Unknown"

live = live_is_online(status)
system_states = int(len(states_npz.get("states", [])))
labels = finite(states_npz.get("labels", [])).reshape(-1) if "labels" in states_npz else np.asarray([])
attack_states = int(np.sum(labels == 1)) if len(labels) else 0
graph_count = load_graph_count(ZEEK_GRAPHS)


# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
<div class="brand-bar">
  <div class="brand-left">
    {icon_html()}
    <div>
      <div class="brand-kicker">AI-POWERED CYBERSECURITY</div>
      <div class="brand-name">Proactive Threat Detection<br>with World Models</div>
      <div class="brand-sub">Predict. Understand. Stay Ahead.</div>
      <div class="brand-tagline">ARJUN transforms network telemetry into a living security state, simulates what may happen next, and maps predicted attack progression to MITRE ATT&amp;CK.</div>
    </div>
  </div>
  <div class="hero-flow">
    <div class="hero-node"><div class="hero-node-icon">☁</div><div><b>Ingest Telemetry</b><span>CIC-IDS · Zeek · Network Logs</span></div></div>
    <div class="hero-node"><div class="hero-node-icon">⌁</div><div><b>Predict Future State</b><span>Graph-conditioned World Model</span></div></div>
    <div class="hero-node"><div class="hero-node-icon">◈</div><div><b>Detect &amp; Prioritize</b><span>Risk Scoring · MITRE Mapping</span></div></div>
    <div class="hero-node"><div class="hero-node-icon">▥</div><div><b>Take Action</b><span>Alerts · Investigation · Response</span></div></div>
    <div class="external-status"><span class="status-dot"></span><span>{'LIVE ZEEK COLLECTOR ONLINE' if live else 'TELEMETRY SNAPSHOT MODE'}</span><span>•</span><span>WORLD MODEL ACTIVE</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STRIP - BELOW HERO, NEVER INSIDE HERO
# ============================================================
st.markdown(
    f"""
<div class="system-strip">
  <div class="system-pill"><div class="s-label">Collector</div><div class="s-value online">● {'ONLINE' if live else 'SNAPSHOT'}</div></div>
  <div class="system-pill"><div class="s-label">Network States</div><div class="s-value cyan">{system_states:,}</div></div>
  <div class="system-pill"><div class="s-label">Graph Records</div><div class="s-value">{graph_count:,}</div></div>
  <div class="system-pill"><div class="s-label">World Model</div><div class="s-value online">● {'ACTIVE' if WORLD_MODEL.exists() else 'MISSING'}</div></div>
  <div class="system-pill"><div class="s-label">Forecast Horizon</div><div class="s-value">{len(future_probs)} STEPS</div></div>
</div>
""",
    unsafe_allow_html=True,
)



# ============================================================
# TOP NAV - DIRECTLY BELOW HERO
# ============================================================
nav_overview, nav_future, nav_whatif, nav_zeek, nav_system = st.tabs([
    "◉ COMMAND CENTER",
    "◈ FUTURE FORECAST",
    "⌁ WHAT-IF",
    "◉ ZEEK SOC",
    "⚙ SYSTEM",
])


# ============================================================
# INGESTION / LIVE ANALYSIS
# ============================================================
UPLOAD_DIR = ROOT / "data" / "raw" / "dashboard_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INGEST_FEATURES = _feature_names() if "_feature_names" in globals() else [
    "flow_count","total_bytes","total_packets","avg_duration",
    "avg_bytes_per_packet","avg_packets_per_second","avg_bytes_per_second",
    "avg_packet_length","packet_length_variance","src_iat_mean",
    "src_iat_variance","dst_iat_mean","dst_iat_variance","iat_mean",
    "iat_variance","syn_count","ack_count","fin_count","rst_count",
    "psh_count","urg_count","avg_tcp_window","tcp_window_variance",
    "ttl_mean","ttl_variance","fragment_flag_count","payload_size_mean",
    "payload_size_variance","retransmission_count","unique_dst_ports",
    "unique_dst_ips","port_scan_count","high_connection_count",
]

def _ingest_num(df, names, default=0.0):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

def _make_state_from_dataframe(df):
    """Convert a CSV/Zeek-like table into ARJUN's 33-D network state."""
    if df is None or df.empty:
        raise ValueError("The uploaded file contains no records.")

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Common aliases.
    aliases = {
        "src_ip":["src_ip","Src IP","Source IP","id.orig_h","source_ip"],
        "dst_ip":["dst_ip","Dst IP","Destination IP","id.resp_h","destination_ip"],
        "src_port":["src_port","Src Port","Source Port","id.orig_p","source_port"],
        "dst_port":["dst_port","Dst Port","Destination Port","id.resp_p","destination_port"],
        "duration":["duration","Duration","Flow Duration"],
        "bytes":["total_bytes","Total Length of Fwd Packets","Total Length of Bwd Packets",
                 "orig_bytes","resp_bytes","bytes","Bytes"],
        "packets":["total_packets","Total Fwd Packets","Total Backward Packets",
                   "orig_pkts","resp_pkts","packets","Packets"],
        "packet_len":["avg_packet_length","Average Packet Size","packet_length","payload_size"],
        "tcp_window":["avg_tcp_window","Init_Win_bytes_forward","tcp_window","window"],
        "ttl":["ttl_mean","TTL","ttl"],
        "syn":["syn_count","SYN Flag Count","flag_syn"],
        "ack":["ack_count","ACK Flag Count","flag_ack"],
        "fin":["fin_count","FIN Flag Count","flag_fin"],
        "rst":["rst_count","RST Flag Count","flag_rst"],
        "psh":["psh_count","PSH Flag Count","flag_psh"],
        "urg":["urg_count","URG Flag Count","flag_urg"],
        "fragment":["fragment_flag_count","fragment_flag","fragment_offset"],
        "retrans":["retransmission_count","retransmission"],
        "payload":["payload_size_mean","payload_size","Payload Length"],
    }

    def col(key, default=0.0):
        return _ingest_num(d, aliases.get(key, [key]), default)

    n = len(d)
    bytes_s = col("bytes")
    packets_s = col("packets")
    dur = col("duration")
    pkt_len = col("packet_len")
    tcpw = col("tcp_window")
    ttl = col("ttl")
    syn = col("syn")
    ack = col("ack")
    fin = col("fin")
    rst = col("rst")
    psh = col("psh")
    urg = col("urg")
    frag = col("fragment")
    retrans = col("retrans")
    payload = col("payload")

    def mean(x):
        return float(x.mean()) if len(x) else 0.0
    def var(x):
        return float(x.var()) if len(x) > 1 else 0.0

    src_iat = np.diff(
        pd.to_numeric(
            d["ts"] if "ts" in d.columns else
            d["Timestamp"] if "Timestamp" in d.columns else
            pd.Series(np.arange(n), index=d.index),
            errors="coerce"
        ).dropna().values
    ) if n > 1 else np.asarray([])
    src_iat = np.asarray(src_iat, dtype=float)
    src_iat = np.abs(src_iat)
    # Zeek timestamps are seconds; other timestamps may be datetime-like.
    if len(src_iat) and np.nanmax(src_iat) > 1000:
        src_iat = src_iat / 1000.0

    src_ips = d[aliases["src_ip"][0]] if aliases["src_ip"][0] in d.columns else pd.Series(dtype=str)
    dst_ips = d[aliases["dst_ip"][0]] if aliases["dst_ip"][0] in d.columns else pd.Series(dtype=str)
    dst_ports = d[aliases["dst_port"][0]] if aliases["dst_port"][0] in d.columns else pd.Series(dtype=str)

    # Direct 33-feature datasets are preferred.
    direct = {}
    lower = {str(c).lower(): c for c in d.columns}
    for f in INGEST_FEATURES:
        if f.lower() in lower:
            direct[f] = mean(pd.to_numeric(d[lower[f.lower()]], errors="coerce").fillna(0.0))

    state = np.zeros(33, dtype=np.float32)
    state[0] = n
    state[1] = mean(bytes_s) if not direct else direct.get("total_bytes", mean(bytes_s))
    state[2] = mean(packets_s) if not direct else direct.get("total_packets", mean(packets_s))
    state[3] = mean(dur)
    state[4] = state[1] / max(state[2], 1.0)
    state[5] = state[2] / max(state[3], 1.0)
    state[6] = state[1] / max(state[3], 1.0)
    state[7] = mean(pkt_len)
    state[8] = var(pkt_len)
    state[9] = mean(src_iat)
    state[10] = var(src_iat)
    state[11] = mean(src_iat)
    state[12] = var(src_iat)
    state[13] = mean(src_iat)
    state[14] = var(src_iat)
    state[15] = float(syn.sum())
    state[16] = float(ack.sum())
    state[17] = float(fin.sum())
    state[18] = float(rst.sum())
    state[19] = float(psh.sum())
    state[20] = float(urg.sum())
    state[21] = mean(tcpw)
    state[22] = var(tcpw)
    state[23] = mean(ttl)
    state[24] = var(ttl)
    state[25] = float(frag.sum())
    state[26] = mean(payload)
    state[27] = var(payload)
    state[28] = float(retrans.sum())
    state[29] = int(pd.to_numeric(dst_ports, errors="coerce").nunique()) if len(dst_ports) else 0
    state[30] = int(dst_ips.astype(str).nunique()) if len(dst_ips) else 0
    state[31] = float(1 if state[29] >= 10 and n >= 10 else 0)
    state[32] = float(1 if n >= 50 else 0)

    # Overlay any exact 33-feature columns present.
    for i, f in enumerate(INGEST_FEATURES):
        if f in direct:
            state[i] = direct[f]

    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _parse_uploaded_file(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        from io import BytesIO
        return pd.read_csv(BytesIO(raw), low_memory=False), "CSV"
    if suffix in {".log", ".txt", ".tsv"}:
        from io import BytesIO
        # Try Zeek TSV first, then generic CSV/whitespace.
        for sep in ("\t", ",", r"\s+"):
            try:
                df = pd.read_csv(BytesIO(raw), sep=sep, engine="python", comment="#")
                if df.shape[1] > 1 and len(df):
                    return df, "LOG"
            except Exception:
                continue
        raise ValueError("Could not parse the uploaded log.")
    if suffix in {".json", ".jsonl"}:
        from io import BytesIO
        try:
            return pd.read_json(BytesIO(raw), lines=True), "LOG"
        except Exception:
            return pd.read_json(BytesIO(raw)), "LOG"
    if suffix in {".pcap", ".pcapng"}:
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, Raw
        except Exception as exc:
            raise RuntimeError("PCAP support requires Scapy. Install it with: pip install scapy") from exc
        import tempfile
        tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
        try:
            tmp.write_bytes(raw)
            packets = rdpcap(str(tmp))
        finally:
            try: tmp.unlink()
            except Exception: pass
        rows = []
        for pkt in packets:
            if IP not in pkt:
                continue
            flags = ""
            window = 0
            tcp_seq = None
            if TCP in pkt:
                flags = str(pkt[TCP].flags)
                window = int(getattr(pkt[TCP], "window", 0) or 0)
                tcp_seq = int(getattr(pkt[TCP], "seq", 0) or 0)
            elif UDP in pkt:
                window = 0
            rows.append({
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "src_port": int(getattr(pkt.payload, "sport", 0) or 0),
                "dst_port": int(getattr(pkt.payload, "dport", 0) or 0),
                "duration": 0.0,
                "bytes": len(pkt),
                "packets": 1,
                "packet_length": len(pkt),
                "tcp_window": window,
                "ttl": int(getattr(pkt[IP], "ttl", 0) or 0),
                "tcp_flags": flags,
                "payload_size": len(bytes(pkt[Raw].payload)) if Raw in pkt else 0,
                "ts": float(pkt.time),
            })
        if not rows:
            raise ValueError("No IPv4 packets were found in the uploaded PCAP.")
        df = pd.DataFrame(rows)
        flags = df["tcp_flags"].astype(str)
        for letter, name in [("S","syn_count"),("A","ack_count"),("F","fin_count"),("R","rst_count"),("P","psh_count"),("U","urg_count")]:
            df[name] = flags.str.contains(letter, regex=False).astype(int)
        return df, "PCAP"
    raise ValueError("Supported formats: .pcap, .pcapng, .csv, .log, .txt, .tsv, .json, .jsonl")

def _score_ingested_state(state):
    if not XGB_MODEL.exists() or joblib is None:
        return None
    artifact = joblib.load(XGB_MODEL)
    model = artifact.get("model", artifact) if isinstance(artifact, dict) else artifact
    normalizer = artifact.get("normalizer") if isinstance(artifact, dict) else None
    threshold_local = float(artifact.get("threshold", 0.55)) if isinstance(artifact, dict) else 0.55
    x = state.reshape(1, -1)
    if x.shape[1] != 33:
        raise RuntimeError(f"Uploaded telemetry produced {x.shape[1]} features; the current XGBoost artifact expects 33.")

    if normalizer is not None:
        if hasattr(normalizer, "transform"):
            x = np.asarray(normalizer.transform(x), dtype=np.float32)
        elif hasattr(normalizer, "scaler") and hasattr(normalizer.scaler, "transform"):
            x = np.asarray(normalizer.scaler.transform(x), dtype=np.float32)
        else:
            raise RuntimeError("The XGBoost normalizer does not expose a usable 33-feature transform.")

    if x.shape[1] != 33:
        raise RuntimeError(f"XGBoost input has {x.shape[1]} features after normalization; expected 33.")

    if hasattr(model, "predict_proba"):
        prob = float(np.asarray(model.predict_proba(x)).reshape(-1)[-1])
    else:
        prob = float(np.asarray(model.predict(x)).reshape(-1)[0])
    return prob, threshold_local

def _run_uploaded_analysis(uploaded):
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    path = UPLOAD_DIR / uploaded.name
    path.write_bytes(raw)
    df, kind = _parse_uploaded_file(uploaded.name, raw)
    state = _make_state_from_dataframe(df)
    result = {
        "filename": uploaded.name,
        "type": kind,
        "records": len(df),
        "state": state,
    }
    scored = _score_ingested_state(state)
    if scored:
        result["attack_probability"], result["threshold"] = scored
    else:
        result["attack_probability"] = None
        result["threshold"] = 0.55
    # Keep the actual uploaded state visible to the dashboard immediately.
    return result

def _render_ingestion():
    """Clean, dark themed file-ingestion entry point for the prototype."""
    st.markdown(
        '<div class="ingest-shell">'
        '<div class="ingest-head">'
        '<div><div class="ingest-title">INGEST NETWORK TELEMETRY</div>'
        '<div class="ingest-sub">Upload a PCAP, Zeek/security log or CSV flow file to run ARJUN analysis.</div>'
        '<div class="ingest-formats">'
        '<span class="ingest-chip">PCAP</span><span class="ingest-chip">PCAPNG</span>'
        '<span class="ingest-chip">CSV</span><span class="ingest-chip">LOG</span>'
        '<span class="ingest-chip">TXT</span><span class="ingest-chip">TSV</span>'
        '<span class="ingest-chip">JSON</span><span class="ingest-chip">JSONL</span>'
        '</div></div>'
        '<div class="ingest-info"><b>MAX 200 MB</b><br>33-D state extraction · Detection · Risk prediction</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Select telemetry file",
        type=["pcap", "pcapng", "csv", "log", "txt", "tsv", "json", "jsonl"],
        key="arjun_ingest_file",
        label_visibility="visible",
    )

    if uploaded is not None:
        st.markdown(
            f'<div class="panel ingest-result" style="margin-top:10px;padding:10px 14px">'
            f'<div class="panel-meta"><b style="color:#EAF9FF">Selected:</b> {uploaded.name} · '
            f'{uploaded.size:,} bytes</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("▶  ANALYZE UPLOADED TELEMETRY", key="analyze_uploaded", use_container_width=True):
            try:
                with st.spinner("ARJUN is ingesting telemetry and building the network state..."):
                    result = _run_uploaded_analysis(uploaded)
                st.session_state["arjun_ingest_result"] = result
                st.success(
                    f"INGESTION COMPLETE · {result['type']} · "
                    f"{result['records']:,} records · 33-D state generated"
                )
            except Exception as exc:
                st.error(f"INGESTION FAILED: {exc}")

    result = st.session_state.get("arjun_ingest_result")
    if result:
        p = result.get("attack_probability")
        if p is not None:
            level = "ATTACK" if p >= result.get("threshold", 0.55) else "BENIGN"
            st.markdown(
                f'<div class="{"danger" if level == "ATTACK" else "safe"}">'
                f'✓ Uploaded telemetry analyzed · Current classifier: <b>{p:.2%}</b> · '
                f'Decision: <b>{level}</b> · threshold {result.get("threshold", 0.55):.0%}</div>',
                unsafe_allow_html=True,
            )
        a, b, c = st.columns(3)
        a.metric("Input Records", f"{result['records']:,}")
        b.metric("State Dimension", "33 features")
        c.metric("Uploaded Type", result["type"])

# Render ingestion before the navigation so judges immediately see the input path.
_render_ingestion()

# ============================================================
# COMMAND CENTER
# ============================================================
with nav_overview:
    if current_prob >= threshold:
        st.markdown('<div class="danger">⚠ CURRENT NETWORK STATE EXCEEDS THE ATTACK DECISION THRESHOLD</div>', unsafe_allow_html=True)
    elif peak_future >= threshold:
        st.markdown('<div class="warn">⚠ FUTURE FORECAST SHOWS ELEVATED ATTACK LIKELIHOOD</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe">✓ NETWORK BELOW ATTACK THRESHOLD — ARJUN PREDICTIVE MONITORING ACTIVE</div>', unsafe_allow_html=True)

    st.markdown("### Security Command Center")
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"Current Attack Score",f"{current_prob:.2%}",f"Decision threshold {threshold:.0%}"),
        (c2,"Peak Future Score",f"{peak_future:.2%}",f"World Model · {len(future_probs)}-step"),
        (c3,"Maximum Risk",f"{peak_risk:.2%}",risk_class(peak_risk)),
        (c4,"Predicted Stage",latest_stage,"MITRE ATT&CK"),
    ]
    for col,label,value,meta in cards:
        with col:
            st.markdown(f'<div class="value-card"><div class="vlabel">{label}</div><div class="vvalue">{value}</div><div class="vmeta">{meta}</div></div>',unsafe_allow_html=True)

    left,right = st.columns([1.5,1])
    with left:
        st.markdown('<div class="section-title">Future Threat Trajectory</div>',unsafe_allow_html=True)
        render_plotly_forecast(future_probs,risk_scores,title="Forecasted attack probability and risk",height=360)
    with right:
        st.markdown('<div class="section-title">System About / Current Evidence</div>',unsafe_allow_html=True)
        st.markdown(f"""
<div class="panel">
<table class="forecast-table">
<tr><td>Network states</td><td>{system_states:,}</td></tr>
<tr><td>Graph records</td><td>{graph_count:,}</td></tr>
<tr><td>State dimension</td><td>33 features</td></tr>
<tr><td>Current attack score</td><td>{current_prob:.2%}</td></tr>
<tr><td>Peak future score</td><td>{peak_future:.2%}</td></tr>
<tr><td>Maximum risk</td><td>{peak_risk:.2%}</td></tr>
<tr><td>Current family</td><td>{current_family}</td></tr>
<tr><td>MITRE stage</td><td>{latest_stage}</td></tr>
</table>
</div>
""",unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecast Timeline</div>',unsafe_allow_html=True)
    render_dark_forecast_table(timeline,future_probs)


# ============================================================
# FUTURE FORECAST
# ============================================================
with nav_future:
    st.markdown("### Future Threat Forecast")
    st.caption("Recursive World Model simulation of future network states.")
    render_plotly_forecast(future_probs,risk_scores,title="Future threat forecast",height=300)

    if len(future_probs):
        rows = []
        for i,p in enumerate(future_probs):
            score = risk_scores[i] if i < len(risk_scores) else 0.0
            item = timeline[i] if i < len(timeline) else {}
            rows.append({"Step":f"t+{i+1}","Attack":f"{p:.2%}","Risk":f"{score:.2%}","Level":str(item.get("risk_level",risk_class(score))).upper(),"MITRE":str(item.get("mitre_stage","Unknown"))})
        render_dark_table(pd.DataFrame(rows))

        st.markdown('<div class="section-title">MITRE ATT&CK Progression</div>',unsafe_allow_html=True)
        stages=[str(x.get("mitre_stage","Unknown")) for x in timeline]
        chips="".join(f'<div class="stage-chip{" active" if i==len(stages) else ""}><span>t+{i}</span><b>{s}</b></div>' for i,s in enumerate(stages,1))
        st.markdown(f'<div class="stage-track">{chips}</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-empty">No forecast artifact found.</div>',unsafe_allow_html=True)


# ============================================================
# WHAT-IF
# ============================================================

def _feature_names():
    return [
        "flow_count", "total_bytes", "total_packets", "avg_duration",
        "avg_bytes_per_packet", "avg_packets_per_second", "avg_bytes_per_second",
        "avg_packet_length", "packet_length_variance", "src_iat_mean",
        "src_iat_variance", "dst_iat_mean", "dst_iat_variance", "iat_mean",
        "iat_variance", "syn_count", "ack_count", "fin_count", "rst_count",
        "psh_count", "urg_count", "avg_tcp_window", "tcp_window_variance",
        "ttl_mean", "ttl_variance", "fragment_flag_count", "payload_size_mean",
        "payload_size_variance", "retransmission_count", "unique_dst_ports",
        "unique_dst_ips", "port_scan_count", "high_connection_count",
    ]


def _unwrap_model_artifact(obj):
    if isinstance(obj, dict):
        return obj.get("model", obj.get("classifier", obj))
    return obj


def _normalizer_feature_count(normalizer):
    """Infer the fitted input width even when sklearn metadata is absent."""
    if normalizer is None:
        return None
    value = getattr(normalizer, "n_features_in_", None)
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    for attr in ("mean_", "scale_", "var_", "center_", "data_min_", "data_max_"):
        value = getattr(normalizer, attr, None)
        if value is not None:
            try:
                return int(np.asarray(value).reshape(-1).shape[0])
            except Exception:
                pass
    value = getattr(normalizer, "feature_names_in_", None)
    if value is not None:
        try:
            return int(len(value))
        except Exception:
            pass
    return None


def _xgb_probability(model_obj, normalizer, raw_state, previous_state=None):
    """Return attack probability using the actual trained XGBoost artifact.

    The supplied improved artifact contains:
      - XGBClassifier with n_features_in_ == 33
      - NetworkStateNormalizer with an internal StandardScaler fitted to 33 features
      - 33 feature names

    Therefore What-If must NOT concatenate state + delta into 66 features.
    It perturbs the 33-D network state, normalizes those 33 values, and sends
    exactly 33 features to the trained classifier.
    """
    x = np.asarray(raw_state, dtype=np.float32).reshape(1, -1)
    if x.shape[1] != 33:
        raise RuntimeError(
            f"XGBoost What-If requires a 33-feature network state; received {x.shape[1]}."
        )

    model = _unwrap_model_artifact(model_obj)
    expected_model_features = getattr(model, "n_features_in_", None)

    # The current production artifact is a 33-feature classifier.
    if expected_model_features is None:
        expected_model_features = 33
    expected_model_features = int(expected_model_features)

    if expected_model_features == 33:
        if normalizer is not None:
            if hasattr(normalizer, "transform"):
                features = np.asarray(normalizer.transform(x), dtype=np.float32)
            elif hasattr(normalizer, "scaler") and hasattr(normalizer.scaler, "transform"):
                features = np.asarray(normalizer.scaler.transform(x), dtype=np.float32)
            else:
                raise RuntimeError(
                    "The XGBoost normalizer does not expose a usable 33-feature transform."
                )
        else:
            # Only used for compatibility with a classifier artifact that was
            # saved without its normalizer.
            features = x
    elif expected_model_features == 66:
        if previous_state is None:
            raise RuntimeError("The loaded XGBoost model expects 66 features, but no previous state was supplied.")
        prev = np.asarray(previous_state, dtype=np.float32).reshape(1, -1)
        if prev.shape[1] != 33:
            raise RuntimeError("Previous state is not 33-dimensional.")
        delta = x - prev
        raw_features = np.concatenate([x, delta], axis=1)
        if normalizer is not None:
            if hasattr(normalizer, "transform"):
                features = np.asarray(normalizer.transform(raw_features), dtype=np.float32)
            elif hasattr(normalizer, "scaler") and hasattr(normalizer.scaler, "transform"):
                scaler = normalizer.scaler
                if getattr(scaler, "n_features_in_", 66) != 66:
                    raise RuntimeError("66-feature XGBoost model has a scaler with a different feature count.")
                features = np.asarray(scaler.transform(raw_features), dtype=np.float32)
            else:
                raise RuntimeError("The 66-feature XGBoost normalizer is not usable.")
        else:
            features = raw_features
    else:
        raise RuntimeError(
            f"Unsupported XGBoost feature count: {expected_model_features}. "
            "Expected 33 or 66."
        )

    if features.shape[1] != expected_model_features:
        raise RuntimeError(
            f"XGBoost expects {expected_model_features} features, "
            f"but What-If produced {features.shape[1]}."
        )

    if hasattr(model, "attack_probability"):
        value = model.attack_probability(features)
        return float(np.asarray(value).reshape(-1)[0])
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(features))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba.reshape(-1)[0])
    if hasattr(model, "predict"):
        return float(np.asarray(model.predict(features)).reshape(-1)[0])

    raise RuntimeError("Unsupported XGBoost artifact.")

def _load_live_what_if_context():
    """Load the actual latest Zeek state history, graph context and models."""
    if not states_npz or "states" not in states_npz:
        raise RuntimeError("Zeek state artifact is unavailable")
    raw_states = np.asarray(states_npz["states"], dtype=np.float32)
    if raw_states.ndim != 2 or raw_states.shape[1] != 33 or len(raw_states) < 2:
        raise RuntimeError("Zeek states do not contain a valid 33-feature history")

    with open(ZEEK_GRAPHS, "rb") as f:
        graphs_obj = pickle.load(f)
    if isinstance(graphs_obj, dict):
        for key in ("graphs", "graph_sequence", "sequence"):
            if key in graphs_obj:
                graphs_obj = graphs_obj[key]
                break
    graphs = list(graphs_obj)
    n = min(10, len(raw_states), len(graphs))
    if n < 2:
        raise RuntimeError("Not enough aligned state/graph history for What-If")

    import torch
    from world_model.hybrid_world_model import HybridWorldModel
    from world_model.hybrid_trainer import StateScaler

    checkpoint = torch.load(WORLD_MODEL, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    state_dim = int(cfg.get("state_dimension", 33))
    graph_dim = int(cfg.get("graph_input_dimension", 6))
    hidden = int(cfg.get("hidden_dimension", 64))
    layers = int(cfg.get("lstm_layers", 1))
    residual_scale = float(cfg.get("residual_scale", 0.75))
    model = HybridWorldModel(
        state_dimension=state_dim,
        graph_input_dimension=graph_dim,
        hidden_dimension=hidden,
        lstm_layers=layers,
        residual_scale=residual_scale,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise RuntimeError("World Model checkpoint has no state dictionary")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    scaler_state = checkpoint.get("state_scaler")
    if not isinstance(scaler_state, dict):
        raise RuntimeError("World Model checkpoint has no state scaler")
    scaler = StateScaler.from_state_dict(scaler_state)

    xgb_artifact = None
    normalizer = None
    if XGB_MODEL.exists():
        # The improved XGBoost artifact is created with joblib.dump().
        # Loading it with pickle.load() causes: invalid load key, '\x0a'.
        if joblib is not None:
            try:
                xgb_artifact = joblib.load(XGB_MODEL)
            except Exception:
                # Backward-compatible fallback for older pickle artifacts.
                with open(XGB_MODEL, "rb") as f:
                    xgb_artifact = pickle.load(f)
        else:
            with open(XGB_MODEL, "rb") as f:
                xgb_artifact = pickle.load(f)
        if isinstance(xgb_artifact, dict):
            normalizer = xgb_artifact.get("normalizer")

    return raw_states[-n:], graphs[-n:], model, scaler, xgb_artifact, normalizer


def _world_model_rollout(raw_history, graph_history, model, scaler, steps=5):
    import torch
    raw_history = np.asarray(raw_history, dtype=np.float32).copy()
    graph_history = list(graph_history)
    current_raw = raw_history[-1].copy()
    raw_out = []
    for _ in range(steps):
        normalized = scaler.transform(raw_history).astype(np.float32)
        state_tensor = torch.from_numpy(normalized)
        with torch.no_grad():
            delta = model(state_tensor, graph_history)
        next_norm = normalized[-1] + delta.detach().cpu().numpy().reshape(-1)
        next_raw = scaler.inverse_transform(next_norm.reshape(1, -1))[0].astype(np.float32)
        next_raw = np.nan_to_num(next_raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw_out.append(next_raw)
        raw_history = np.vstack([raw_history[1:], next_raw])
        # The current implementation does not predict future topology, so the
        # latest observed graph is held constant during recursive rollout.
        graph_history = graph_history[1:] + [graph_history[-1]]
        current_raw = next_raw
    return np.asarray(raw_out, dtype=np.float32)


with nav_whatif:
    st.markdown("### What-If Analysis")
    st.caption("Live counterfactual analysis from the latest 10 Zeek states. Each selected feature is perturbed, rolled forward through the World Model, and scored with the trained XGBoost classifier.")

    live_wf_rows = []
    live_wf_error = None
    try:
        history, graph_history, wf_model, wf_scaler, wf_xgb, wf_normalizer = _load_live_what_if_context()
        names = _feature_names()
        if isinstance(wf_xgb, dict) and isinstance(wf_xgb.get("feature_names"), (list, tuple)):
            # The current improved production classifier uses a 33-feature raw network state.
            # Keep the first 33 names for the raw network state.
            artifact_names = list(wf_xgb["feature_names"])
            if len(artifact_names) >= 33:
                names = artifact_names[:33]

        baseline_states = _world_model_rollout(history, graph_history, wf_model, wf_scaler, steps=5)
        baseline_prev = history[-1]
        baseline_probs = []
        if wf_xgb is not None:
            prev = baseline_prev
            for state in baseline_states:
                baseline_probs.append(_xgb_probability(wf_xgb, wf_normalizer, state, prev))
                prev = state
        else:
            baseline_probs = list(future_probs[:5])

        # Use the features with the largest current magnitude/change first.
        current = history[-1]
        delta = current - history[-2]
        ranking = np.argsort(-(np.abs(current) + 0.5 * np.abs(delta)))
        selected = ranking[:8]

        for idx in selected:
            feature = names[idx] if idx < len(names) else f"feature_{idx}"
            original = float(current[idx])
            # For zero-valued counters, use a one-unit perturbation so the
            # analysis is not mathematically identical at +/-25% of zero.
            base_amount = abs(original) if abs(original) > 1e-9 else 1.0
            for pct in (-25.0, 25.0):
                cf_history = history.copy()
                if original >= 0:
                    cf_history[-1, idx] = original + (base_amount * pct / 100.0)
                    cf_history[-1, idx] = max(0.0, cf_history[-1, idx])
                else:
                    cf_history[-1, idx] = original * (1.0 + pct / 100.0)
                cf_states = _world_model_rollout(cf_history, graph_history, wf_model, wf_scaler, steps=5)
                cf_probs = []
                if wf_xgb is not None:
                    prev = cf_history[-1]
                    for state in cf_states:
                        cf_probs.append(_xgb_probability(wf_xgb, wf_normalizer, state, prev))
                        prev = state
                else:
                    cf_probs = list(baseline_probs)
                b = float(baseline_probs[-1]) if baseline_probs else 0.0
                c = float(cf_probs[-1]) if cf_probs else b
                live_wf_rows.append({
                    "feature": feature,
                    "original_value": original,
                    "perturbation_percent": pct,
                    "counterfactual_value": float(cf_history[-1, idx]),
                    "baseline_t+5_attack_probability": b,
                    "counterfactual_t+5_attack_probability": c,
                    "probability_change": c - b,
                })
    except Exception as exc:
        live_wf_error = str(exc)

    if live_wf_rows:
        frame = pd.DataFrame(live_wf_rows)
        frame["original_value"] = frame["original_value"].map(lambda x: f"{x:.3f}")
        frame["counterfactual_value"] = frame["counterfactual_value"].map(lambda x: f"{x:.3f}")
        frame["perturbation_percent"] = frame["perturbation_percent"].map(lambda x: f"{x:+.0f}%")
        for col in ("baseline_t+5_attack_probability", "counterfactual_t+5_attack_probability", "probability_change"):
            frame[col] = frame[col].map(lambda x: f"{x:.2%}")
        render_dark_table(frame)
        st.markdown('<div class="recommendation"><div class="recommendation-title">Model-based interpretation</div><p>Each row is generated by perturbing one feature in the latest observed Zeek state, recursively rolling the World Model forward for five steps, and comparing the resulting XGBoost attack probability with the unmodified trajectory.</p></div>', unsafe_allow_html=True)
    elif live_wf_error:
        st.markdown(
            f'<div class="warn">Live What-If could not be computed from the current model artifact: {live_wf_error}. '
            'No stale sensitivity values are shown.</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown('<div class="alert-empty">No What-If result is available.</div>', unsafe_allow_html=True)


# ============================================================
# ZEEK SOC + ALERTS
# ============================================================

# ============================================================
with nav_zeek:
    st.markdown("### Zeek Predictive SOC")
    st.caption("Zeek telemetry → 33-D state → World Model → future risk → MITRE → predictive alert.")

    r1,r2,r3,r4 = st.columns([1.1,1,1,1.35])
    with r1:
        if st.button("↻ Refresh Telemetry",key="refresh_zeek"):
            st.rerun()
    with r2:
        st.metric("Alert records",f"{len(alerts):,}")
    with r3:
        st.metric("Unacknowledged",f"{sum(not bool(a.get('acknowledged',False)) for a in alerts):,}")
    with r4:
        if st.button("🚨 Create Predictive Alert", key="create_predictive_alert"):
            if AlertEngine is None:
                st.error("Alert engine module could not be loaded.")
            else:
                try:
                    engine = AlertEngine(ZEEK_RISK_JSON, ALERT_HISTORY)
                    alert = engine.raise_alert()
                    if alert is not None:
                        st.success(
                            f"Predictive alert created: {alert.alert_id} "
                            f"· {alert.severity} · t+{alert.forecast_step}"
                        )
                        st.rerun()
                    else:
                        st.info(
                            "No new alert: the current forecast is already "
                            "recorded or no forecast is available."
                        )
                except Exception as exc:
                    st.error(f"Alert creation failed: {exc}")


    if live:
        st.markdown('<div class="safe">● LIVE ZEEK COLLECTOR ONLINE — telemetry heartbeat is fresh.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">● ZEEK LIVE COLLECTOR NOT CONFIRMED — showing latest processed telemetry.</div>',unsafe_allow_html=True)

    z1,z2,z3,z4 = st.columns(4)
    z1.metric("Zeek states",f"{system_states:,}")
    z2.metric("Graph records",f"{graph_count:,}")
    z3.metric("Peak forecast",f"{peak_future:.2%}")
    z4.metric("Peak risk",f"{peak_risk:.2%}")

    # Prominent latest-alert card using persisted alert history.
    if alerts:
        latest_alert = alerts[-1]
        p = safe_float(latest_alert.get("attack_probability", latest_alert.get("probability", 0.0)))
        r = safe_float(latest_alert.get("risk_score", 0.0))
        stage = str(latest_alert.get("mitre_stage", "Unknown"))
        sev = str(latest_alert.get("severity", risk_class(r))).upper()
        status = "ACKNOWLEDGED" if bool(latest_alert.get("acknowledged", False)) else "ACTIVE"
        st.markdown(
            f'''<div class="live-alert-card">
<div class="live-alert-top"><b>ARJUN PREDICTIVE ALERT</b><span class="table-badge {sev.lower()}">{sev}</span><span class="live-alert-status">{status}</span></div>
<div class="live-alert-grid">
<div><small>Alert ID</small><strong>{str(latest_alert.get("alert_id","—"))}</strong></div>
<div><small>Attack probability</small><strong>{p:.2%}</strong></div>
<div><small>Risk score</small><strong>{r:.2%}</strong></div>
<div><small>MITRE stage</small><strong>{stage}</strong></div>
</div></div>''',
            unsafe_allow_html=True
        )

    email_configured = all([
        bool(__import__("os").getenv("ARJUN_EMAIL_HOST")),
        bool(__import__("os").getenv("ARJUN_EMAIL_USERNAME")),
        bool(__import__("os").getenv("ARJUN_EMAIL_PASSWORD")),
        bool(__import__("os").getenv("ARJUN_ALERT_RECIPIENT")),
    ])
    st.markdown(
        f'<div class="panel-meta">Alert engine: ACTIVE · '
        f'Email notification: {"CONFIGURED" if email_configured else "NOT CONFIGURED"} · '
        f'History: {len(alerts):,} record(s)</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">Live Predictive Alerts</div>',unsafe_allow_html=True)
    if alerts:
        for idx in range(len(alerts)-1,-1,-1):
            item=alerts[idx]
            sev=alert_severity(item); badge=alert_badge_class(sev); ack=bool(item.get("acknowledged",False))
            prob=safe_float(item.get("attack_probability",0)); rs=safe_float(item.get("risk_score",0)); step=item.get("forecast_step","—")
            stage=str(item.get("mitre_stage",item.get("stage","Unknown"))); aid=str(item.get("alert_id","ARJUN-PREDICTED")); ts=alert_time(item.get("timestamp",""))
            status_text="ACKNOWLEDGED" if ack else str(item.get("status","PREDICTED")).upper()
            indicators=[]
            for x in item.get("indicators",[])[:5] if isinstance(item.get("indicators",[]),list) else []:
                indicators.append(str(x.get("name",x.get("indicator",x))) if isinstance(x,dict) else str(x))
            indicator_text=", ".join(indicators) or "Forecast-driven predictive alert"
            recommendation=str(item.get("recommendation","Investigate the predicted stage and correlate with live telemetry."))
            st.markdown(f"""
<div class="alert-center">
  <div class="alert-row">
    <div><div class="alert-id">🚨 {aid}</div><div class="alert-time">{ts}</div></div>
    <div><span class="badge {badge}">{sev}</span></div>
    <div><span class="badge badge-predicted">{status_text}</span></div>
    <div class="alert-action"><b>Attack</b><br>{prob:.2%}</div>
    <div class="alert-action"><b>Risk</b><br>{rs:.2%} · t+{step}</div>
    <div class="alert-action"><b>MITRE</b><br>{stage}</div>
  </div>
  <div class="alert-action" style="margin:8px 8px 0"><b>Indicators:</b> {indicator_text}<br><b>Recommendation:</b> {recommendation}</div>
</div>
""",unsafe_allow_html=True)
            if not ack:
                if st.button("✓ Acknowledge Alert",key=f"ack_{aid}_{idx}"):
                    try:
                        if AlertEngine is not None:
                            AlertEngine(ZEEK_RISK_JSON, ALERT_HISTORY).acknowledge(aid)
                        else:
                            alerts[idx]["acknowledged"] = True
                            alerts[idx]["status"] = "ACKNOWLEDGED"
                            save_alert_history(alerts)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not acknowledge alert: {exc}")
    else:
        st.markdown('<div class="alert-empty">No predictive alerts are currently recorded. When the alert engine creates one, it will appear here with severity, probability, risk, MITRE stage and recommended action.</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Forecast-to-Alert Timeline</div>',unsafe_allow_html=True)
    render_dark_forecast_table(timeline,future_probs)
    st.markdown('<div class="recommendation"><div class="recommendation-title">SOC workflow</div><p>ARJUN generates predictive alerts from forecasted attack probability and risk, attaches the predicted MITRE stage and evidence, then presents the alert for SOC investigation and acknowledgement.</p></div>',unsafe_allow_html=True)


# ============================================================
# SYSTEM
# ============================================================
with nav_system:
    st.markdown("### System & About ARJUN")
    st.markdown("""
<div class="panel">
  <div class="panel-title">ARJUN — AI-Driven Risk Forecasting for Proactive Cyber Defence</div>
  <div class="panel-meta" style="font-size:14px;line-height:1.7">
    ARJUN moves from network visibility to future threat prediction. It converts network traffic and Zeek telemetry
    into a structured 33-feature network state, combines it with host-to-host communication graphs and temporal context,
    and uses learned state-transition dynamics to forecast what may happen next.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### What has been completed")
    completed = [
        ("01", "CIC-IDS-2018 dataset preparation", "Processed CIC-IDS-2018 traffic into a chronological dataset of 56,700 network states with 33 features."),
        ("02", "Feature engineering", "Implemented flow, packet, temporal, behavioural, contextual and graph-oriented features."),
        ("03", "Network state + graph representation", "Represented observations as a 33-D state vector with communication-graph information and temporal context."),
        ("04", "Attack classification", "Trained and integrated the XGBoost layer for current network-state attack classification."),
        ("05", "ARJUN World Model", "Trained the Temporal GNN + LSTM World Model to learn P(Sₜ₊₁ | Sₜ, Gₜ) transition dynamics."),
        ("06", "World Model validation", "Validated next-state forecasting against persistence and tested generalization on the held-out Infilteration family."),
        ("07", "K-step forecasting", "Implemented recursive multi-step rollout to generate future network states and attack trajectories."),
        ("08", "Risk + MITRE intelligence", "Mapped forecast outputs to risk scores, risk levels and predicted MITRE ATT&CK stages."),
        ("09", "Zeek integration", "Connected Zeek telemetry to the state → graph → World Model → risk → MITRE pipeline and verified new telemetry can trigger an alert."),
        ("10", "Alert + SOC prototype", "Implemented predictive alert generation, alert history and acknowledgement; email remains optional."),
        ("11", "Interactive dashboard", "Built Command Center, Future Forecast, What-If, Zeek SOC, ingestion and System views."),
    ]
    for num, title, desc in completed:
        st.markdown(f"""
<div class="panel" style="margin:7px 0;padding:12px 16px;display:flex;gap:14px;align-items:flex-start">
  <div style="min-width:38px;color:#19D8FF;font-weight:900;font-size:14px">{num}</div>
  <div><div style="color:#F3FAFF;font-weight:900;font-size:14px">{title}</div>
  <div style="color:#9FC0D6;font-size:12px;line-height:1.5;margin-top:3px">{desc}</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Current prototype status")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("CIC-IDS-2018 states", f"{system_states:,}")
    c2.metric("State features", "33")
    c3.metric("World Model", "TRAINED" if WORLD_MODEL.exists() else "MISSING")
    c4.metric("Zeek states", f"{system_states:,}")
    c5.metric("Forecast", f"{len(future_probs)} steps")

    st.markdown("#### What happens next")
    next_steps = [
        ("01", "Upload / live telemetry", "Use PCAP, PCAPNG, Zeek logs and CSV/JSON telemetry as new network input and convert it into ARJUN states."),
        ("02", "Temporal anomaly detection", "Add the Temporal LSTM Autoencoder from the target architecture to learn normal sequential behaviour and strengthen anomaly detection."),
        ("03", "Future attack graph", "Extend the rollout to forecast future host-to-host graph structure, likely targets and attack paths."),
        ("04", "Explainable forecasting", "Strengthen temporal explanations and What-If analysis so SOC users can identify influential features and time steps."),
        ("05", "Operational response", "Connect predictive alerts to configurable notification and recommended-mitigation workflows; email can be enabled when required."),
        ("06", "Broader validation", "Evaluate additional attack families and datasets, calibrate probabilities and benchmark the complete forecasting pipeline."),
    ]
    for num, title, desc in next_steps:
        st.markdown(f"""
<div class="panel" style="margin:7px 0;padding:12px 16px;display:flex;gap:14px;align-items:flex-start">
  <div style="min-width:38px;color:#62E9FF;font-weight:900;font-size:14px">{num}</div>
  <div><div style="color:#F3FAFF;font-weight:900;font-size:14px">{title}</div>
  <div style="color:#9FC0D6;font-size:12px;line-height:1.5;margin-top:3px">{desc}</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="recommendation" style="margin-top:16px">
  <div class="recommendation-title">ARJUN pipeline</div>
  <p style="margin:6px 0 0">Network Telemetry → Preprocessing → Feature Engineering → 33-D Network State + Graph → AI/ML Intelligence → World Model → K-Step Future States → Risk + MITRE ATT&CK → Explainability → Predictive Alerts → SOC Dashboard.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;margin-top:35px;color:#6F92AC;font-size:12px">ARJUN • AI-Based Network Attack Forecasting • Offline SOC Prototype</div>',unsafe_allow_html=True)

