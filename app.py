import io
import json
import hashlib
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from streamlit_js_eval import get_geolocation
except Exception:
    get_geolocation = None

try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
except Exception:
    A4 = colors = getSampleStyleSheet = ParagraphStyle = TA_CENTER = None
    SimpleDocTemplate = Paragraph = Spacer = Table = TableStyle = RLImage = None


APP = "SIH 2026 — Smart Field Test Analyzer"
ROOT = Path(__file__).resolve().parent
EMBLEM_PATH = ROOT / "assets" / "indian_emblem_logo.png"
DB = ROOT / "prototype_records.db"
MODEL_DIR = ROOT / "models"

KIT_FILES = {
    "Standard Narcotic Drugs Detection Kit": "standard_narcotics.joblib",
    "Precursor Chemicals Detection Kit": "precursor_chemicals.joblib",
    "Ketamine Detection Kit": "ketamine.joblib",
}

FEATURE_COLUMNS = [
    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    "lab_l_mean", "lab_l_std", "lab_a_mean", "lab_a_std",
    "lab_b_mean", "lab_b_std", "brightness", "contrast",
    "blur_score", "glare_ratio",
]

st.set_page_config(
    page_title=APP,
    page_icon=str(EMBLEM_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Visual design ----------------
st.markdown(
    """
    <style>
    :root {
        --ink: #1b2633;
        --muted: #5f6b78;
        --line: #cfd6dd;
        --surface: #f4f6f8;
        --surface-2: #e9edf1;
        --navy: #315b78;
        --navy-dark: #24455c;
        --green: #216a4a;
        --red: #9f3028;
        --amber: #8a6500;
    }
    .stApp { background: #e7ebef; color: var(--ink); }
    [data-testid="stSidebar"] { background: #dce2e8; border-right: 1px solid #c5cdd5; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    .block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--ink); }
    h1, h2, h3 { letter-spacing: -0.02em; }

    /* Streamlit form controls: deliberately dark-text and low-glare. */
    div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {
        background: #f8fafb !important;
        border: 1px solid #b9c3cd !important;
        border-radius: 9px !important;
    }
    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        color: #1b2633 !important;
        background: transparent !important;
        -webkit-text-fill-color: #1b2633 !important;
    }
    div[data-baseweb="select"] * { color: #1b2633 !important; }
    input::placeholder, textarea::placeholder { color: #7b8794 !important; opacity: 1 !important; }
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #344250 !important; font-weight: 650 !important; }
    [data-testid="stCaptionContainer"] p { color: #65717e !important; }

    /* Buttons */
    .stButton > button {
        border: 1px solid #aeb9c4 !important;
        background: #f2f4f6 !important;
        color: #1b2633 !important;
        border-radius: 9px !important;
        font-weight: 650 !important;
    }
    .stButton > button:hover { background: #e4e9ee !important; border-color: #8e9aa7 !important; }
    .stButton > button[kind="primary"] {
        background: #315b78 !important; border-color: #315b78 !important; color: #ffffff !important;
    }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span { color: #ffffff !important; }
    [data-testid="stDownloadButton"] button { background: #315b78 !important; color: #ffffff !important; border-color: #315b78 !important; }
    [data-testid="stDownloadButton"] button * { color: #ffffff !important; }

    /* Tabs and radio navigation */
    button[data-baseweb="tab"] { color: #344250 !important; font-weight: 650 !important; }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span { color: #344250 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #315b78 !important; }
    [data-testid="stSidebar"] label p, [data-testid="stSidebar"] label span { color: #263542 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label { background: transparent !important; border-radius: 8px; padding: 5px 8px; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #cfd7df !important; }

    .brand { padding: 0.2rem 0 1.1rem 0; }
    .brand-mark {
        display:inline-flex; width:42px; height:42px; border-radius:10px;
        align-items:center; justify-content:center; background:#eef1f4;
        border:1px solid #c5ced7; font-size:22px; margin-right:10px;
    }
    .brand-title { font-size:20px; font-weight:750; color:var(--ink) !important; vertical-align:middle; }
    .brand-sub { color:var(--muted) !important; font-size:12px; margin-top:5px; }
    .hero {
        background: #f1f3f5; border:1px solid var(--line); border-radius:16px;
        padding:24px 26px; margin-bottom:18px; box-shadow:0 5px 16px rgba(31,43,55,.05);
    }
    .hero-kicker { color:var(--navy) !important; font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .hero-title { color:var(--ink) !important; font-size:32px; font-weight:800; margin:4px 0 6px; }
    .hero-text { color:var(--muted) !important; font-size:15px; max-width:850px; line-height:1.55; }
    .section-card {
        background:#f2f4f6; border:1px solid var(--line); border-radius:14px;
        padding:18px 20px; margin-bottom:16px; box-shadow:0 3px 12px rgba(31,43,55,.035);
    }
    .section-title { color:var(--ink) !important; font-weight:750; font-size:17px; margin-bottom:4px; }
    .section-help { color:var(--muted) !important; font-size:13px; margin-bottom:12px; }
    .metric-card {
        background:#f2f4f6; border:1px solid var(--line); border-radius:12px; padding:16px 18px;
        box-shadow:0 3px 10px rgba(31,43,55,.03);
    }
    .metric-label { color:var(--muted) !important; font-size:12px; font-weight:650; }
    .metric-value { color:var(--ink) !important; font-size:25px; font-weight:800; margin-top:3px; }
    .result-positive { background:#f7e9e7; border:1px solid #dfbbb6; color:var(--red); }
    .result-negative { background:#e8f2ed; border:1px solid #bfd7c9; color:var(--green); }
    .result-inconclusive { background:#f4efdf; border:1px solid #ddcf9d; color:var(--amber); }
    .result-box { border-radius:14px; padding:20px; text-align:center; margin:8px 0 16px; }
    .result-label { font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; opacity:.85; }
    .result-value { font-size:30px; font-weight:850; margin-top:2px; }
    .info-row { display:flex; justify-content:space-between; gap:20px; padding:10px 0; border-bottom:1px solid #d9dee3; }
    .info-row:last-child { border-bottom:0; }
    .info-key { color:var(--muted) !important; font-size:13px; }
    .info-val { color:var(--ink) !important; font-size:13px; font-weight:650; text-align:right; }
    .status-pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#dfe5ea; color:#344250 !important; font-size:12px; font-weight:700; }
    .footer-note { color:#66727e !important; font-size:12px; text-align:center; padding-top:22px; }
    .auth-shell { max-width:760px; margin:4vh auto 0; }
    .auth-card { background:#eef1f4; border:1px solid #cbd3da; border-radius:18px; padding:30px; box-shadow:0 8px 24px rgba(31,43,55,.07); }
    .small-muted { color:var(--muted) !important; font-size:13px; }
    div[data-testid="stMetric"] { background:#f1f3f5; border:1px solid var(--line); padding:12px 14px; border-radius:12px; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#5f6b78 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#1b2633 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Database ----------------
def db():
    return sqlite3.connect(DB)


def init_db():
    con = db()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS records(
            case_id TEXT PRIMARY KEY,
            operator_id TEXT,
            kit_type TEXT,
            result TEXT,
            confidence REAL,
            timestamp_utc TEXT,
            latitude REAL,
            longitude REAL,
            address TEXT,
            location_status TEXT,
            image_sha256 TEXT,
            signature TEXT,
            model_version TEXT,
            reason TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_utc TEXT NOT NULL
        )
        """
    )
    # Lightweight migration for older prototype databases.
    existing = {r[1] for r in con.execute("PRAGMA table_info(records)").fetchall()}
    additions = {
        "address": "TEXT", "location_status": "TEXT", "signature": "TEXT",
        "model_version": "TEXT", "reason": "TEXT", "image_sha256": "TEXT",
    }
    for col, typ in additions.items():
        if col not in existing:
            con.execute(f"ALTER TABLE records ADD COLUMN {col} {typ}")
    con.commit()
    con.close()


init_db()


# ---------------- Authentication ----------------
def hash_password(password, salt=None):
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt_bytes, 200_000)
    return salt_bytes.hex(), digest.hex()


def create_account(email, password):
    email = email.strip().lower()
    salt, digest = hash_password(password)
    con = db()
    try:
        con.execute(
            "INSERT INTO users(email,password_hash,salt,created_utc) VALUES(?,?,?,?)",
            (email, digest, salt, datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        con.close()


def authenticate(email, password):
    email = email.strip().lower()
    con = db()
    row = con.execute("SELECT password_hash,salt FROM users WHERE email=?", (email,)).fetchone()
    con.close()
    if not row:
        return False
    _, digest = hash_password(password, row[1])
    return hmac.compare_digest(digest, row[0])


def require_auth():
    return st.session_state.get("auth_mode") in {"account", "guest"}


def auth_screen():
    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.image(str(EMBLEM_PATH), width=110)
    st.markdown(
        '<h1 style="text-align:center;margin-bottom:4px">SIH 2026 Smart Field Test Analyzer</h1>'
        '<p class="small-muted" style="text-align:center">Secure, user-friendly presumptive field-test documentation</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Sign in", "➕ Create account"])
    with tab1:
        st.markdown("### Welcome back")
        email = st.text_input("Email", placeholder="name@example.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in", type="primary", use_container_width=True):
            if not email or not password:
                st.error("Enter both email and password.")
            elif authenticate(email, password):
                st.session_state.auth_mode = "account"
                st.session_state.user_email = email.strip().lower()
                st.rerun()
            else:
                st.error("Incorrect email or password.")
    with tab2:
        st.markdown("### Create your account")
        email2 = st.text_input("Email", placeholder="name@example.com", key="create_email")
        p1 = st.text_input("Password", type="password", key="create_password")
        p2 = st.text_input("Confirm password", type="password", key="create_confirm")
        if st.button("Create account", type="primary", use_container_width=True):
            if not email2 or "@" not in email2:
                st.error("Enter a valid email address.")
            elif len(p1) < 6:
                st.error("Password should contain at least 6 characters for this prototype.")
            elif p1 != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_account(email2, p1)
                if ok:
                    st.session_state.auth_mode = "account"
                    st.session_state.user_email = email2.strip().lower()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown("**Prefer to explore first?**")
    st.caption("You can continue without an account. Account sign-in is useful when you want your operator identity attached to records.")
    if st.button("Continue without signing in", use_container_width=True):
        st.session_state.auth_mode = "guest"
        st.session_state.user_email = "Guest"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if not require_auth():
    auth_screen()
    st.stop()


# ---------------- Security / record integrity ----------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def canonical_image_bytes(image):
    """Return a stable PNG representation for hashing/verification."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def sign_record(r):
    payload = {
        "case_id": r["case_id"],
        "operator_id": r["operator_id"],
        "timestamp_utc": r["timestamp_utc"],
        "kit_type": r["kit_type"],
        "result": r["result"],
        "image_sha256": r["image_sha256"],
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "address": r["address"],
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


# ---------------- Colour model ----------------
@st.cache_resource
def load_detector():
    """Load the optional YOLO detector used to find the reaction area."""
    if YOLO is None:
        return None
    path = MODEL_DIR / "yolo_best.pt"
    if not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception:
        return None


def detect_reaction_roi(image):
    """
    Find the reaction area in the captured photo.

    The classifier was trained on reaction-area colour features. A fixed
    image-center crop works for the supplied demo images, but phone photos
    can be portrait-oriented or framed differently. YOLO therefore finds
    the reaction area first and the colour features are extracted from that
    detected region.
    """
    detector = load_detector()
    if detector is None:
        return None, 0.0

    try:
        rgb = np.asarray(image.convert("RGB"))
        results = detector.predict(source=rgb, conf=0.20, iou=0.45, verbose=False)
        if not results:
            return None, 0.0

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        best = None
        best_conf = 0.0
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            # YOLO class 1 = reaction_area
            if cls_id == 1 and conf > best_conf:
                best = box.xyxy[0].cpu().numpy().tolist()
                best_conf = conf

        if best is None:
            return None, 0.0

        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in best]
        x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        crop = image.crop((x1, y1, x2, y2))
        return crop, best_conf
    except Exception:
        return None, 0.0


def _colour_features_from_array(a, center=(0.50, 0.50), radii=(0.23, 0.35)):
    h, w = a.shape[:2]

    # The detected reaction crop uses its central area. The full-image
    # fallback uses the original demo-dataset ROI proportions.
    cx, cy = w * center[0], h * center[1]
    rx, ry = max(4.0, w * radii[0]), max(4.0, h * radii[1])
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    # Fallback for unusually small crops.
    if int(mask.sum()) < 25:
        mask = np.ones((h, w), dtype=bool)

    pix = a[mask].astype(np.float32)
    f = {}
    for i, n in enumerate(["r", "g", "b"]):
        f[f"{n}_mean"] = float(pix[:, i].mean())
        f[f"{n}_std"] = float(pix[:, i].std())

    if cv2 is not None:
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)[mask]
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)[mask]
        for i, n in enumerate(["h", "s", "v"]):
            f[f"{n}_mean"] = float(hsv[:, i].mean())
            f[f"{n}_std"] = float(hsv[:, i].std())
        for i, n in enumerate(["lab_l", "lab_a", "lab_b"]):
            f[f"{n}_mean"] = float(lab[:, i].mean())
            f[f"{n}_std"] = float(lab[:, i].std())
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gp = gray[mask]
        f["brightness"] = float(gp.mean())
        f["contrast"] = float(gp.std())
        f["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        f["glare_ratio"] = float(np.mean(gp >= 245))
    else:
        gray = np.dot(a.astype(np.float32), [0.299, 0.587, 0.114])
        gp = gray[mask]
        for n in ["h", "s", "v", "lab_l", "lab_a", "lab_b"]:
            f[f"{n}_mean"] = 0.0
            f[f"{n}_std"] = 0.0
        f["brightness"] = float(gp.mean())
        f["contrast"] = float(gp.std())
        f["blur_score"] = 0.0
        f["glare_ratio"] = float(np.mean(gp >= 245))
    return f


def extract_features(image, use_detector=True):
    """
    Extract the same 22-feature colour vector used by the trained RF models.

    For phone captures, prefer a YOLO-detected reaction region. If YOLO is
    unavailable or cannot find the region, fall back to the original
    proportional ROI used for the supplied demo dataset.
    """
    if use_detector:
        reaction_crop, _ = detect_reaction_roi(image)
        if reaction_crop is not None:
            return _colour_features_from_array(np.asarray(reaction_crop.convert("RGB")))

    return _colour_features_from_array(
        np.asarray(image.convert("RGB")),
        center=(0.594, 0.500),
        radii=(0.106, 0.142),
    )

def quality_score(f):
    score = (
        0.25 * (1 - min(abs(f["brightness"] - 128) / 128, 1))
        + 0.25 * min(f["contrast"] / 70, 1)
        + 0.30 * min(f["blur_score"] / 150, 1)
        + 0.20 * max(0, 1 - f["glare_ratio"] * 5)
    )
    return float(np.clip(score, 0, 1))


@st.cache_resource
def load_model(path):
    if joblib is None or not Path(path).exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def run_model(features, kit):
    path = MODEL_DIR / KIT_FILES[kit]
    package = load_model(path)
    if package is None:
        return None, 0.0
    try:
        if isinstance(package, dict) and "model" in package:
            estimator = package["model"]
            cols = package.get("feature_columns", FEATURE_COLUMNS)
        else:
            estimator = package
            cols = getattr(estimator, "feature_names_in_", FEATURE_COLUMNS)
        row = {c: float(features.get(c, 0.0)) for c in cols}
        x = pd.DataFrame([row], columns=list(cols))
        raw = str(estimator.predict(x)[0]).strip().lower()
        mapping = {
            "positive": "Positive", "negative": "Negative", "inconclusive": "Inconclusive",
            "valid_positive": "Positive", "valid_negative": "Negative", "valid_inconclusive": "Inconclusive",
        }
        label = mapping.get(raw, str(raw).title())
        confidence = float(np.max(estimator.predict_proba(x)[0])) if hasattr(estimator, "predict_proba") else 0.0
        return label, confidence
    except Exception:
        return None, 0.0


# ---------------- Location ----------------
def get_location_silently():
    if get_geolocation is None:
        return None, "Location component unavailable"
    try:
        loc = get_geolocation()
        if not loc:
            return None, "Waiting for browser location permission"
        if "error" in loc:
            return None, loc["error"].get("message", "Browser location unavailable")
        coords = loc.get("coords", {})
        if coords.get("latitude") is None or coords.get("longitude") is None:
            return None, "Browser location unavailable"
        return {
            "latitude": float(coords["latitude"]),
            "longitude": float(coords["longitude"]),
            "accuracy": coords.get("accuracy"),
        }, "GPS captured"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def reverse_geocode(lat, lon):
    if Nominatim is None:
        return ""
    try:
        geocoder = Nominatim(user_agent="sih2026-smart-field-test-analyzer")
        result = geocoder.reverse((lat, lon), zoom=18, language="en", exactly_one=True)
        return result.address if result else ""
    except Exception:
        return ""


# ---------------- Analysis + storage ----------------
def analyze(image, kit):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_bytes = canonical_image_bytes(image)
    timestamp = datetime.now(timezone.utc).isoformat()
    image_hash = sha256_bytes(image_bytes)
    reaction_crop, detector_conf = detect_reaction_roi(image)
    if reaction_crop is not None:
        features = _colour_features_from_array(np.asarray(reaction_crop.convert("RGB")))
        region_found = True
    else:
        features = extract_features(image, use_detector=False)
        region_found = False

    q = quality_score(features)
    pred, conf = run_model(features, kit)

    if pred is None:
        result, confidence = "Inconclusive", 0.0
        reason = "The dataset-trained colour model could not be loaded."
    elif conf < 0.50:
        result, confidence = "Inconclusive", conf
        reason = "The captured reaction colour is below the prototype confidence threshold; retake the image with the reaction area clearly visible and evenly lit."
    elif pred == "Inconclusive":
        result, confidence = "Inconclusive", conf
        reason = "The reaction colour matches the dataset's inconclusive class."
    else:
        result, confidence = pred, conf
        if region_found:
            reason = "Result classified from the detected reaction area using RGB/HSV/Lab colour features and the supplied dataset-trained model."
        else:
            reason = "Result classified from reaction-colour RGB/HSV/Lab features using the supplied dataset-trained model."

    return {
        "case_id": "", "operator_id": "", "kit_type": kit,
        "result": result, "confidence": confidence, "timestamp_utc": timestamp,
        "latitude": None, "longitude": None, "address": "", "location_status": "",
        "image_sha256": image_hash, "signature": "", "model_version": "reaction-color-v3",
        "reason": reason, "quality_score": q,
    }


def save_record(r):
    con = db()
    con.execute(
        """
        INSERT OR REPLACE INTO records(
            case_id,operator_id,kit_type,result,confidence,timestamp_utc,
            latitude,longitude,address,location_status,image_sha256,signature,
            model_version,reason
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            r["case_id"], r["operator_id"], r["kit_type"], r["result"], r["confidence"],
            r["timestamp_utc"], r["latitude"], r["longitude"], r["address"],
            r["location_status"], r["image_sha256"], r["signature"], r["model_version"], r["reason"],
        ),
    )
    con.commit()
    con.close()


# ---------------- PDF ----------------
def make_pdf(r, image):
    if SimpleDocTemplate is None:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=16)
    result_style = ParagraphStyle("Result", parent=styles["Heading1"], alignment=TA_CENTER, spaceAfter=14)
    story = [
        Paragraph("SIH 2026 — Smart Field Test Analysis Report", title),
        Paragraph(f"Result: <b>{r['result'].upper()}</b>", result_style),
        Paragraph("Presumptive screening result — not a laboratory confirmation.", styles["Normal"]),
        Spacer(1, 12),
    ]
    rows = [
        ["Case ID", r["case_id"]], ["Operator ID", r["operator_id"]], ["Kit type", r["kit_type"]],
        ["Result", r["result"]], ["Confidence", f"{r['confidence'] * 100:.1f}%"],
        ["Image quality", f"{r['quality_score'] * 100:.1f}%"], ["UTC timestamp", r["timestamp_utc"]],
        ["Location", r["address"] or ("GPS coordinates captured" if r["latitude"] is not None and r["longitude"] is not None else "Not available")],
        ["Latitude", f"{r['latitude']:.6f}" if r["latitude"] is not None else "Not available"],
        ["Longitude", f"{r['longitude']:.6f}" if r["longitude"] is not None else "Not available"],
        ["Location coordinates", f"{r['latitude']:.6f}, {r['longitude']:.6f}" if r["latitude"] is not None and r["longitude"] is not None else "Not available"],
        ["Location status", r["location_status"] or "Not available"],
        ["Decision reason", r["reason"]], ["SHA-256", r["image_sha256"]], ["Record signature", r["signature"]],
    ]
    table = Table(rows, colWidths=[125, 390])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    if r["latitude"] is not None and r["longitude"] is not None:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"GPS captured: {r['latitude']:.6f}, {r['longitude']:.6f}",
            styles["Normal"]
        ))
    story.append(Spacer(1, 16))
    try:
        img_buf = io.BytesIO()
        report_image = image.copy()
        report_image.thumbnail((450, 300))
        report_image.save(img_buf, format="PNG")
        img_buf.seek(0)
        story.append(Paragraph("Captured image", styles["Heading2"]))
        story.append(RLImage(img_buf, width=420, height=280))
    except Exception:
        pass
    story += [
        Spacer(1, 12),
        Paragraph(
            "Prototype note: the classifier was trained on the supplied synthetic/demo dataset. "
            "Production use requires authorised kit-specific reference data, validation, security review, and laboratory correlation.",
            styles["Normal"],
        ),
    ]
    doc.build(story)
    return buffer.getvalue()


# ---------------- App shell ----------------
def logout():
    for key in ["auth_mode", "user_email", "last", "last_image", "last_pdf"]:
        st.session_state.pop(key, None)
    st.rerun()


NAV_PAGES = [
    "🏠 Dashboard",
    "📷 New Test",
    "🗂️ History",
    "🔐 Verify Image",
    "ℹ️ How It Works",
    "⚙️ System Status",
]

# Quick-action buttons select the destination before the next rerun.
# The sidebar radio uses the same session-state value, so the requested
# workspace opens immediately instead of showing a navigation hint.
if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "🏠 Dashboard"
if st.session_state.get("go_page"):
    st.session_state["page_nav"] = st.session_state.pop("go_page")

with st.sidebar:
    st.image(str(EMBLEM_PATH), width=58)
    st.markdown(
        '<div class="brand"><span class="brand-title">SIH 2026</span>'
        '<div class="brand-sub">Smart Field Test Analyzer</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Signed in as: {st.session_state.get('user_email', 'Guest')}")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        NAV_PAGES,
        key="page_nav",
        label_visibility="visible",
    )
    st.divider()
    if st.button("↪ Sign out", use_container_width=True):
        logout()
    st.caption("Presumptive screening prototype")


# ---------------- Dashboard ----------------
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Smart India Hackathon 2026</div>'
        '<div class="hero-title">Smart Field Test Analyzer</div>'
        '<div class="hero-text">A mobile-friendly workflow for capturing field-test results, '
        'classifying reaction colour, recording case context and generating an integrity-protected PDF report.</div></div>',
        unsafe_allow_html=True,
    )
    con = db()
    df = pd.read_sql_query("SELECT * FROM records ORDER BY timestamp_utc DESC", con)
    con.close()
    total = len(df)
    positives = int((df["result"] == "Positive").sum()) if not df.empty else 0
    negatives = int((df["result"] == "Negative").sum()) if not df.empty else 0
    inconclusive = int((df["result"] == "Inconclusive").sum()) if not df.empty else 0
    a, b, c, d = st.columns(4)
    a.metric("🧾 Total cases", total)
    b.metric("🔴 Positive", positives)
    c.metric("🟢 Negative", negatives)
    d.metric("🟠 Inconclusive", inconclusive)
    st.markdown("### Quick actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown('<div class="section-card"><div class="section-title">📷 Start a new test</div><div class="section-help">Capture or upload a test image and create a case record.</div></div>', unsafe_allow_html=True)
        if st.button("📷 Open New Test", use_container_width=True):
            st.session_state["go_page"] = "📷 New Test"
            st.rerun()
    with q2:
        st.markdown('<div class="section-card"><div class="section-title">🗂️ Review history</div><div class="section-help">Search previous cases and export the audit list.</div></div>', unsafe_allow_html=True)
        if st.button("🗂️ Open History", use_container_width=True):
            st.session_state["go_page"] = "🗂️ History"
            st.rerun()
    with q3:
        st.markdown('<div class="section-card"><div class="section-title">🔐 Verify an image</div><div class="section-help">Use a case ID and the original image to check integrity.</div></div>', unsafe_allow_html=True)
        if st.button("🔐 Verify Image", use_container_width=True):
            st.session_state["go_page"] = "🔐 Verify Image"
            st.rerun()
    st.warning("⚠️ Presumptive result only — laboratory confirmation is required for evidentiary use.")


# ---------------- New test ----------------
elif page == "📷 New Test":
    st.markdown("<div class='hero'><div class='hero-kicker'>Field workflow</div><div class='hero-title'>New Test</div><div class='hero-text'>Capture the complete test setup. Location is requested only after an image is available.</div></div>", unsafe_allow_html=True)
    if st.button("🧹 Start a fresh test", key="fresh_test"):
        for key in ["last", "last_image", "last_pdf"]:
            st.session_state.pop(key, None)
        st.rerun()

    with st.container(border=True):
        st.markdown("### 1. Case details")
        c1, c2 = st.columns(2)
        with c1:
            case = st.text_input("Case ID", f"CASE-{uuid.uuid4().hex[:8].upper()}")
        with c2:
            default_operator = st.session_state.get("user_email", "Guest")
            operator = st.text_input("Operator ID", default_operator)
        kit = st.selectbox("Drug-Test Kit", list(KIT_FILES), help="Select the kit profile used for this field test.")

    with st.container(border=True):
        st.markdown("### 2. Capture evidence image")
        source = st.radio("Image source", ["📱 Camera", "📁 Upload image"], horizontal=True, label_visibility="collapsed")
        image = None
        if source == "📱 Camera":
            st.caption("📱 For phone camera access, open this app over HTTPS. A local network URL such as http://192.168.x.x:8501 may be blocked by the mobile browser.")
            cam = st.camera_input("Capture the complete test image")
            if cam:
                image = ImageOps.exif_transpose(Image.open(cam)).convert("RGB")
                st.success("📸 Image captured successfully")
                st.caption("Tip: keep the complete test card and reaction area inside the frame, avoid glare, and hold the phone steady. If the result is Inconclusive, retake a closer, evenly lit image.")
        else:
            up = st.file_uploader("Upload the complete test image", type=["jpg", "jpeg", "png", "webp"])
            if up:
                image = ImageOps.exif_transpose(Image.open(up)).convert("RGB")

    if image:
        st.image(image, caption="Captured test image", use_container_width=True)

        # GPS request starts only after a photo exists.
        location, location_status = get_location_silently()
        lat = location["latitude"] if location else None
        lon = location["longitude"] if location else None
        address = reverse_geocode(lat, lon) if location else ""

        with st.container(border=True):
            st.markdown("### 3. Location")
            if location:
                st.success("📍 Location captured")
                loc_cols = st.columns(3)
                loc_cols[0].metric("Latitude", f"{lat:.6f}")
                loc_cols[1].metric("Longitude", f"{lon:.6f}")
                loc_cols[2].metric("GPS status", "Captured")
                if address:
                    st.caption(f"📍 {address}")
            else:
                st.warning("📍 GPS location is not available yet. Allow location permission in the browser and click **Refresh location**.")
                if st.button("📍 Refresh location", key="refresh_location", use_container_width=True):
                    st.rerun()

        if st.button("🔎 Analyze Test", type="primary", use_container_width=True):
            with st.spinner("Analysing reaction colour and preparing the record…"):
                r = analyze(image, kit)
                r.update(case_id=case.strip(), operator_id=operator.strip(), latitude=lat, longitude=lon, address=address, location_status=location_status)
                r["signature"] = sign_record(r)
                save_record(r)
                st.session_state["last"] = r
                st.session_state["last_image"] = image
                st.session_state["last_pdf"] = make_pdf(r, image)
            st.success("Analysis complete. Record saved.")

    if "last" in st.session_state:
        r = st.session_state["last"]
        st.divider()
        st.markdown("## Result")
        result_class = {"Positive": "result-positive", "Negative": "result-negative", "Inconclusive": "result-inconclusive"}[r["result"]]
        st.markdown(f'<div class="result-box {result_class}"><div class="result-label">Presumptive result</div><div class="result-value">{r["result"]}</div></div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Confidence", f"{r['confidence'] * 100:.1f}%")
        m2.metric("🖼️ Image quality", f"{r['quality_score'] * 100:.1f}%")
        m3.metric("🕒 UTC recorded", r["timestamp_utc"].replace("+00:00", " UTC"))

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card"><div class="section-title">🧾 Case information</div>', unsafe_allow_html=True)
            for k, v in [("Case ID", r["case_id"]), ("Operator", r["operator_id"]), ("Kit", r["kit_type"]), ("Decision", r["reason"])]:
                st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card"><div class="section-title">📍 Location</div>', unsafe_allow_html=True)
            if r["latitude"] is not None and r["longitude"] is not None:
                st.markdown(f'<div class="info-row"><span class="info-key">Address</span><span class="info-val">{r["address"] or "Address lookup unavailable"}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-row"><span class="info-key">Coordinates</span><span class="info-val">{r["latitude"]:.6f}, {r["longitude"]:.6f}</span></div>', unsafe_allow_html=True)
                try:
                    st.map(pd.DataFrame({"lat": [r["latitude"]], "lon": [r["longitude"]]}), zoom=15)
                except Exception:
                    pass
            else:
                st.markdown(f'<div class="info-row"><span class="info-key">Status</span><span class="info-val">{r["location_status"]}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">📄 Official-style analysis report</div><div class="section-help">The PDF contains the complete integrity record, including the SHA-256 value. The hash is intentionally hidden from this frontend.</div></div>', unsafe_allow_html=True)
        pdf = st.session_state.get("last_pdf")
        if pdf:
            st.download_button("⬇️ Download Analysis Report (PDF)", data=pdf, file_name=f"{r['case_id']}_analysis_report.pdf", mime="application/pdf", type="primary", use_container_width=True)

        st.caption("🔒 Integrity data is retained in the record and PDF for verification; it is not displayed on the result screen.")
        st.warning("⚠️ Presumptive screening result only. Do not treat this prototype result as laboratory confirmation.")


# ---------------- History ----------------
elif page == "🗂️ History":
    st.markdown("<div class='hero'><div class='hero-kicker'>Audit trail</div><div class='hero-title'>Case History</div><div class='hero-text'>Search, review and export recorded field-test cases.</div></div>", unsafe_allow_html=True)
    con = db()
    df = pd.read_sql_query("SELECT case_id,operator_id,kit_type,result,confidence,timestamp_utc,address,location_status FROM records ORDER BY timestamp_utc DESC", con)
    con.close()
    if not df.empty:
        s1, s2, s3 = st.columns([2, 1, 1])
        with s1:
            search = st.text_input("🔎 Search", placeholder="Case ID, operator, kit, result or location")
        with s2:
            result_filter = st.selectbox("Result", ["All", "Positive", "Negative", "Inconclusive"])
        with s3:
            kit_filter = st.selectbox("Kit", ["All"] + list(KIT_FILES.keys()))
        if search:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)]
        if result_filter != "All":
            df = df[df["result"] == result_filter]
        if kit_filter != "All":
            df = df[df["kit_type"] == kit_filter]
        if not df.empty:
            df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Export filtered history (CSV)", df.to_csv(index=False).encode("utf-8"), "sih_2026_history.csv", "text/csv")
        else:
            st.info("No cases match the selected filters.")
    else:
        st.info("No records yet. Start a new test to create the first case.")


# ---------------- Verification ----------------
elif page == "🔐 Verify Image":
    st.markdown("<div class='hero'><div class='hero-kicker'>Integrity check</div><div class='hero-title'>Verify Original Image</div><div class='hero-text'>Enter the case ID and upload the original image used for that record. The integrity hash is checked in the background and is never shown on this screen.</div></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Verify a case")
        st.caption("Only the Case ID and the original image are required. Integrity values remain hidden.")
        case = st.text_input("Case ID", placeholder="CASE-XXXXXXXX", help="Use the case ID from the analysis report.")
        up = st.file_uploader("Upload original image", type=["jpg", "jpeg", "png", "webp"], key="verify")
        if st.button("🔎 Verify image", type="primary", use_container_width=True):
            if not case.strip():
                st.error("Enter a Case ID.")
            elif not up:
                st.error("Upload the original image.")
            else:
                con = db()
                row = con.execute("SELECT image_sha256 FROM records WHERE case_id=?", (case.strip(),)).fetchone()
                con.close()
                if not row:
                    st.error("Case ID not found.")
                else:
                    try:
                        uploaded_image = Image.open(up)
                        current = sha256_bytes(canonical_image_bytes(uploaded_image))
                    except Exception:
                        current = ""
                    if hmac.compare_digest(current, row[0]):
                        st.success("✅ VERIFIED — the uploaded image matches the stored record.")
                        st.markdown('<div class="section-card"><div class="section-title">Integrity status</div><div class="section-help">The image content is unchanged from the file associated with this case.</div></div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ NOT VERIFIED — the uploaded image does not match the stored record.")
                        st.markdown('<div class="section-card"><div class="section-title">Integrity status</div><div class="section-help">The uploaded file differs from the original recorded image.</div></div>', unsafe_allow_html=True)


# ---------------- How it works ----------------
elif page == "ℹ️ How It Works":
    st.markdown("<div class='hero'><div class='hero-kicker'>Prototype workflow</div><div class='hero-title'>How the Analyzer Works</div><div class='hero-text'>A simple field workflow designed for a jury to understand in seconds.</div></div>", unsafe_allow_html=True)
    steps = [
        ("01", "📷 Capture", "Take a clear photo of the complete field-test setup or upload a captured image."),
        ("02", "🎨 Analyse colour", "The selected kit profile uses RGB, HSV and Lab colour features from the reaction region."),
        ("03", "📊 Classify", "A dataset-trained model returns Positive, Negative or Inconclusive with a confidence score."),
        ("04", "📍 Record context", "UTC time, operator/case information and browser GPS are attached when available."),
        ("05", "📄 Generate report", "A PDF report is generated with the analysis, image and integrity information."),
        ("06", "🔐 Verify", "Later, a reviewer can enter only the Case ID and upload the original image to check integrity."),
    ]
    cols = st.columns(2)
    for i, (num, title, text) in enumerate(steps):
        with cols[i % 2]:
            st.markdown(f'<div class="section-card"><div style="font-size:12px;font-weight:800;color:#667085">STEP {num}</div><div class="section-title">{title}</div><div class="section-help">{text}</div></div>', unsafe_allow_html=True)
    st.info("Prototype boundary: the supplied training data is synthetic/demo data. Real deployment requires authorised kit-specific reference data, validation and laboratory correlation.")


# ---------------- Status ----------------
elif page == "⚙️ System Status":
    st.markdown("<div class='hero'><div class='hero-kicker'>Readiness</div><div class='hero-title'>System Status</div><div class='hero-text'>A compact health check for the prototype components.</div></div>", unsafe_allow_html=True)
    components = [
        ("🖥️ Streamlit UI", True), ("🗃️ SQLite records", DB.exists()), ("🧠 Joblib", joblib is not None),
        ("🖼️ OpenCV", cv2 is not None), ("📍 Browser GPS", get_geolocation is not None),
        ("🗺️ Address lookup", Nominatim is not None), ("📄 PDF reports", SimpleDocTemplate is not None),
    ]
    for kit, filename in KIT_FILES.items():
        components.append((f"🧪 {kit}", (MODEL_DIR / filename).exists()))
    rows = pd.DataFrame({"Component": [x[0] for x in components], "Status": ["Ready" if x[1] else "Unavailable" for x in components]})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.warning("⚠️ Prototype only: presumptive screening does not replace laboratory confirmation.")

st.markdown('<div class="footer-note">SIH 2026 • Smart Field Test Analyzer • Presumptive screening prototype</div>', unsafe_allow_html=True)
