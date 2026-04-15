import base64
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
import tempfile
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection System",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }

    /* Card used for each detected person */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        border: 3px solid rgba(255,255,255,0.3);
        margin-bottom: 10px;
    }

    /* Summary bar at the top of results section */
    .summary-bar {
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        background-color: #1e2130;
        border: 2px solid #444;
        margin-bottom: 16px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGO HEADER
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open(r"E:\petroChoise\Task\Helmet-Detection-with-colors\images\logo.png", "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>
        .header {{
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 0px 0px;
        }}
        .header img {{ width: 100px; }}
        .company-name {{ font-size: 25px; font-weight: bold; }}
        .subtitle {{ font-size: 20px; color: white; }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{logo_data}">
            <div>
                <div class="company-name">PETROCHOISE</div>
                <div class="subtitle">INTEGRATED SERVICES</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 35px; font-weight: bold;">PETROCHOISE</div>
            <div style="font-size: 20px; color: gray;">INTEGRATED SERVICES</div>
        </div>
    """, unsafe_allow_html=True)
    
st.title("Helmet Detection System")
st.markdown("### Real-Time Multi-Helmet & Color Detection")

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND IMAGE
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open(r"E:\petroChoise\Task\Helmet-Detection-with-colors\images\Photorealistic_high-quality_construction_area_insi-1776079121348.png", "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# COLOR DETECTION — get_dominant_color
# Uses KMeans clustering on the HSV values of the helmet crop.
# Focuses on the top-center region where the dome of the helmet sits.
# ─────────────────────────────────────────────────────────────────────────────
def get_dominant_color(image_region, k=3):
    """
    Extracts the dominant color of a helmet crop using KMeans on HSV pixels.
    Returns (color_name: str, hsv_tuple: tuple).
    Falls back to 'Unknown' if the region is too small or flat.
    """
    if image_region is None or image_region.size == 0:
        return "Unknown", (0, 0, 0)

    # Crop to top-center dome area to avoid background / face noise
    h, w = image_region.shape[:2]
    crop = image_region[int(h * 0.10):int(h * 0.40),
                        int(w * 0.35):int(w * 0.65)]

    # Blur to reduce pixel noise before clustering
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    # Convert BGR → HSV for perceptually meaningful clustering
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Remove low-saturation pixels (gray/white background noise)
    pixels = hsv.reshape(-1, 3)
    mask   = pixels[:, 1] > 40        # saturation threshold
    pixels = pixels[mask]

    # Fallback: use all pixels if saturation filter removed too many
    if len(pixels) < 50:
        pixels = hsv.reshape(-1, 3)

    # KMeans clustering — find the most common color cluster
    kmeans  = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)

    counts   = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    h_val, s_val, v_val = dominant

    color_name = classify_color_hsv(h_val, s_val, v_val)
    return color_name, (int(h_val), int(s_val), int(v_val))


# ─────────────────────────────────────────────────────────────────────────────
# COLOR CLASSIFICATION — classify_color_hsv
# Maps an HSV triplet to a human-readable color name.
# ─────────────────────────────────────────────────────────────────────────────
def classify_color_hsv(h, s, v):
    """
    Rule-based HSV → color name mapping.
    Handles dark (black), low-saturation (white/gray), then hue bands.
    """
    if v < 50:
        return "Black"

    if s < 40:
        return "White" if v > 200 else "White"

    # Hue bands (OpenCV HSV: H in [0, 179])
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"

    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION FUNCTION — detect_helmet_and_color  (MODIFIED for multi-class)
#
# CHANGES FROM ORIGINAL:
#   • Loops over ALL boxes in the frame (was: only best_idx)
#   • Reads class_id per box to distinguish helmet vs no-helmet
#   • helmet class   → runs color extraction → color name
#   • no-helmet class → skips color extraction → color = "N/A"
#   • Returns a LIST of detection dicts + annotated frame
#     instead of (bool, str, float, frame)
#
# Each dict in the returned list:
#   {
#     "label":      "Helmet" | "No Helmet",
#     "color":      "Red" | "Blue" | ... | "Unknown" | "N/A",
#     "confidence": float (0–1),
#     "box":        (x1, y1, x2, y2)
#   }
# ─────────────────────────────────────────────────────────────────────────────

HELMET_CLASS_ID    = 0   # "helmet"
NO_HELMET_CLASS_ID = 1   # "no helmet"

def detect_helmet_and_color(frame, model):
    results = model(frame, conf=0.5, verbose=False)

    detections      = []           # collects one dict per detected person
    annotated_frame = frame.copy()

    # BGR colors used to draw bounding boxes per helmet color
    helmet_color_map = {
        "Red":     (0, 0, 255),
        "Blue":    (255, 0, 0),
        "Green":   (0, 255, 0),
        "Yellow":  (0, 255, 255),
        "Orange":  (0, 165, 255),
        "White":   (255, 255, 255),
        "Black":   (50, 50, 50),
        "Gray":    (128, 128, 128),
        "Unknown": (200, 200, 200),
    }

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # ── Iterate over EVERY detected box (multi-person support) ────
        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id   = int(boxes.cls[i].cpu().numpy())   # 0=helmet, 1=no-helmet
            box        = boxes.xyxy[i].cpu().numpy().astype(int)

            x1, y1, x2, y2 = box
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # w = x2 - x1
            # h = y2 - y1

            # # ── Face-filter: very tall-narrow boxes are faces, not helmets ──
            # ratio = h / max(w, 1)
            # if ratio > 1.15:
            #     continue

            # ── Branch on detected class ─────────────────────────────────
            if class_id == HELMET_CLASS_ID:
                label         = "Helmet"
                helmet_region = frame[y1:y2, x1:x2]

                # Run color extraction only for the helmet class
                if helmet_region.size > 0:
                    helmet_color, _ = get_dominant_color(helmet_region)
                else:
                    helmet_color = "Unknown"

                # Box color matches the detected helmet color
                box_color = helmet_color_map.get(helmet_color, (0, 255, 0))

            elif class_id == NO_HELMET_CLASS_ID:
                # No physical helmet → no color to extract
                label        = "No Helmet"
                helmet_color = "N/A"
                box_color    = (0, 0, 255)   # always red — safety violation

            else:
                # Unknown class id — skip gracefully
                continue

            # ── Draw annotated bounding box ──────────────────────────────
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

            # Build label string: show color only when it exists
            if helmet_color != "N/A":
                display_label = f"{label}: {helmet_color} ({confidence:.2f})"
            else:
                display_label = f"{label} ({confidence:.2f})"

            cv2.putText(
                annotated_frame,
                display_label,
                (x1, max(y1 - 10, 15)),   # clamp above-frame labels
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

            # ── Store detection for Streamlit results panel ───────────────
            detections.append({
                "label":      label,
                "color":      helmet_color,
                "confidence": confidence,
                "box":        (x1, y1, x2, y2)
            })

    # Returns empty list if no detection passed filters — caller handles this
    return detections, annotated_frame


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DISPLAY — show_all_results  (REPLACES show_result)
#
# CHANGES FROM ORIGINAL:
#   • Accepts a list of detections instead of single values
#   • Renders a summary bar (total workers / helmets / violations)
#   • Renders one result card per detected person, side-by-side
#   • Shows "No Detection" card if the list is empty
# ─────────────────────────────────────────────────────────────────────────────
def show_all_results(detections):
    """
    Renders detection results in Streamlit.

    3 cases:
      1. Empty list       → grey "No Detection" card
      2. Only no-helmet   → red cards per person, summary shows 0 helmets
      3. Mixed / helmets  → green cards with color, red cards for violations
    """

    # ── Case 1: model ran but nothing passed filters ─────────────────────
    # DE tmamm
    if not detections:
        st.markdown(
            """
            <div class="result-box" style="background-color:#CC3333; color:white;">
                 <b>NO DETECTION</b><br>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # ── Summary counts ────────────────────────────────────────────────────
    total      = len(detections)
    helmets    = sum(1 for d in detections if d["label"] == "Helmet")
    violations = total - helmets

    # Color-code the summary bar by safety status
    if violations == 0:
        summary_color = "#00AA00"   # all safe
    else :
        summary_color = "#CC3333"   # all unsafe
    # de tmamm
    st.markdown(
        f"""
        <div class="summary-bar" style="border-color:{summary_color};">
             With Helmet: <b>{helmets}</b> &nbsp;|&nbsp;
             Without Helmet: <b>{violations}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── One card per detected person (side-by-side columns) ──────────────
    cols = st.columns(max(total, 1))

    for idx, det in enumerate(detections):
        with cols[idx]:

            if det["label"] == "Helmet":
                bg    = "#00AA00"
                icon  = " "
                title = "HELMET DETECTED"
            else:
                bg    = "#CC3333"
                icon  = " "
                title = "NO HELMET"

            # Show color line only when color was extracted
            if det["color"] != "N/A":
                color_line = f"Color: <b>{det['color']}</b>"
            else:
                color_line = "Color: <b>N/A</b>"

            st.markdown(
                f"""
                <div class="result-box" style="background-color:{bg}; color:white;">
                    <span style="font-size:15px;">{title}</span><br>
                    {color_line}<br>
                    Confidence: <b>{det['confidence']:.2%}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    YOLO_MODEL_PATH = "best (4).pt"  
    model = YOLO(YOLO_MODEL_PATH)
    return model    
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.error("Make sure 'best (4).pt' is in the same directory as this script.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🎥 Real-Time Camera", use_container_width=True, key="camera_btn"):
        st.session_state.mode = "camera"

with col2:
    if st.button("📤 Upload Image / Video", use_container_width=True, key="upload_btn"):
        st.session_state.mode = "upload"

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA MODE — Real-time multi-person detection
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.mode == "camera":

    stop_button      = st.button("⏹ Stop Camera", key="stop_camera")
    image_placeholder  = st.empty()
    result_placeholder = st.empty()

    # camera index 0 = default webcam; change to 1 for external camera
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Check your camera connection or change the camera index.")
    else:
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera.")
                break

            # ── Run multi-person detection ────────────────────────────────
            detections, annotated_frame = detect_helmet_and_color(frame, model)

            # ── Display annotated video frame ─────────────────────────────
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            # ── Render result cards below the frame ───────────────────────
            with result_placeholder.container():
                show_all_results(detections)

        cap.release()
        st.info("Camera stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD MODE — Image or Video
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.mode == "upload":

    file = st.file_uploader(
        "📤 Upload Image or Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if file:
        file_type          = file.type
        image_placeholder  = st.empty()
        result_placeholder = st.empty()

        # ── IMAGE ──────────────────────────────────────────────────────────
        if "image" in file_type:
            image    = Image.open(file)
            image_np = np.array(image)

            # PIL gives RGB; convert to BGR for OpenCV/YOLO
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np

            # ── Run multi-person detection ────────────────────────────────
            detections, annotated_frame = detect_helmet_and_color(image_bgr, model)

            # Convert back to RGB for Streamlit display
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(
                annotated_rgb,
                caption=f"Detection Result — {len(detections)} person(s) found",
                use_container_width=True
            )

            with result_placeholder.container():
                show_all_results(detections)

        # ── VIDEO ──────────────────────────────────────────────────────────
        elif "video" in file_type:

            # Write uploaded bytes to a temp file so OpenCV can open it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file.getbuffer())
                temp_video_path = tmp.name

            cap          = cv2.VideoCapture(temp_video_path)
            fps          = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            stop_video             = st.button("⏹ Stop Video", key="stop_video")
            frame_placeholder      = st.empty()
            result_placeholder_vid = st.empty()
            progress_bar           = st.progress(0)

            frame_count = 0

            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Run detection on every frame ──────────────────────────
                detections, annotated_frame = detect_helmet_and_color(frame, model)

                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, use_container_width=True)

                with result_placeholder_vid.container():
                    show_all_results(detections)

                # Update progress bar
                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)

                frame_count += 1

            progress_bar.progress(1.0)
            cap.release()
            st.success("✅ Video playback complete!")

            # Clean up temp file
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass

st.markdown("---")