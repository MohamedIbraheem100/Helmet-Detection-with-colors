import base64
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import os
from sklearn.cluster import KMeans
import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Helmet Detection System",
    layout="wide"
)

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ===============================
# CUSTOM CSS 
# ===============================
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
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        border: 3px solid white;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOGO AND HEADER
# ===============================
# Only load logo if file exists
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
        .header img {{
            width: 100px;
        }}
        .company-name {{
            font-size: 25px;
            font-weight: bold;
        }}
        .subtitle {{
            font-size: 20px;
            color: white;
        }}
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
 

# ===============================
# HEADER
# ===============================
st.title("Helmet Detection System")
st.markdown("### Detecting helmets and identifying their colors")

# ===============================
# BACKGROUND IMAGE
# ===============================
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

# ===============================
# COLOR DETECTION FUNCTIONS
# ===============================
def get_dominant_color(image_region, k=3):
    
    if image_region is None or image_region.size == 0:
        return "Unknown", (0, 0, 0)

    # 🔹 Step 1: Center crop (focus on helmet)### me7taga tetzbat aw n run model a7sn fel IOU
    h, w = image_region.shape[:2]
    crop = image_region[int(h*0.10):int(h*0.40), int(w*0.35):int(w*0.65)]

    # 🔹 Step 2: Blur to reduce noise
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    # 🔹 Step 3: Convert to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 🔹 Step 4: Remove low saturation (gray/white noise)
    pixels = hsv.reshape(-1, 3)
    mask = pixels[:,1] > 40   # saturation filter
    pixels = pixels[mask]

    if len(pixels) < 50:
        pixels = hsv.reshape(-1, 3)

    # 🔹 Step 5: KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get dominant cluster
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    h, s, v = dominant

    color_name = classify_color_hsv(h, s, v)

    return color_name, (int(h), int(s), int(v))

def classify_color_hsv(h, s, v):
    """
    More robust color classification using HSV
    """

    if v < 50:
        return "Black"

    if s < 40:
        if v > 200:
            return "White"
        return "Gray"

    # Hue-based classification
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

# ===============================
# DETECTION FUNCTION
# ===============================
def detect_helmet_and_color(frame, model):
    results = model(frame, verbose=False)

    helmet_detected = False
    helmet_color = "None"
    confidence = 0.0
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        confidence = float(confidences[best_idx])
        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        w = x2 - x1
        h = y2 - y1
        ratio = h / max(w, 1)

        if ratio > 0.9:  #filter faces to prevent wrong detection y3m
            continue
        
        helmet_region = frame[y1:y2, x1:x2]

        if helmet_region.size > 0:
            helmet_detected = True
            helmet_color, _ = get_dominant_color(helmet_region)

            # 🎨 Color-aware bounding box
            color_map = {
                "Red": (0, 0, 255),
                "Blue": (255, 0, 0),
                "Green": (0, 255, 0),
                "Yellow": (0, 255, 255),
                "Orange": (0, 165, 255),
                "White": (255, 255, 255),
                "Black": (50, 50, 50),
                "Gray": (128, 128, 128),
            }

            box_color = color_map.get(helmet_color, (0, 255, 0))

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

            label = f"{helmet_color} ({confidence:.2f})"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

    return helmet_detected, helmet_color, confidence, annotated_frame

# ===============================
# RESULT DISPLAY FUNCTION
# ===============================
def show_result(helmet_detected, helmet_color, confidence):
    """Display detection results with appropriate styling"""
    if helmet_detected:
        bg_color = "#00AA00"
        text_color = "white"
        text = f"✅ <span style='color: {text_color};'>HELMET DETECTED</span><br>Color: <b>{helmet_color}</b><br>Confidence: <b>{confidence:.2%}</b>"
    else:
        bg_color = "#CC3333"
        text_color = "white"
        text = f"❌ <span style='color: {text_color};'>NO HELMET DETECTED</span><br>Color: <b>N/A</b>"

    st.markdown(
        f"""
        <div class="result-box" style="background-color:{bg_color}; color:{text_color};">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        # Try to load from common paths
        model_paths = [
            "best (4).pt",
            "./best (4).pt",
            "/home/user/best (4).pt",
            # Add your actual path here
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return YOLO(path)
        
        # If not found, try direct path
        return YOLO("best (4).pt")
    except Exception as e:
        raise e

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.error("Make sure 'best.pt' is in the same directory as this script")
    st.stop()

# ===============================
# MODE SELECTION
# ===============================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Real-Time Camera", use_container_width=True, key="camera_btn"):
        st.session_state.mode = "camera"

with col2:
    if st.button("Upload Image / Video", use_container_width=True, key="upload_btn"):
        st.session_state.mode = "upload"

# ===============================
# CAMERA MODE
# ===============================
if st.session_state.mode == "camera":
    st.info("Starting Camera... Press 'Stop' to end the stream.")
    
    # Add stop button
    stop_button = st.button("Stop Camera", key="stop_camera")
    
    image_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Try different camera indices
    cap = None
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            # st.success(f"Camera {1} opened successfully!")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
    else:
        frame_count = 0
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            # Run detection every frame
            helmet_detected, helmet_color, confidence, annotated_frame = detect_helmet_and_color(frame, model)
            
            # Convert BGR to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            image_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Show result
            with result_placeholder.container():
                show_result(helmet_detected, helmet_color, confidence)
        
        cap.release()
        st.info("Camera stopped.")

# ===============================
# UPLOAD MODE
# ===============================
elif st.session_state.mode == "upload":
    file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if file:
        file_type = file.type
        
        image_placeholder = st.empty()
        result_placeholder = st.empty()

        if "image" in file_type:
            # Read image
            image = Image.open(file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Run detection
            helmet_detected, helmet_color, confidence, annotated_frame = detect_helmet_and_color(image_bgr, model)
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            image_placeholder.image(annotated_frame_rgb, caption="Detection Result", use_container_width=True)
            
            with result_placeholder.container():
                show_result(helmet_detected, helmet_color, confidence)

        elif "video" in file_type:
    
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(file.getbuffer())
                temp_video_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
           
            # Add stop button
            stop_video = st.button("Stop Video", key="stop_video")
            
            frame_placeholder = st.empty()
            result_placeholder_video = st.empty()
            progress_bar = st.progress(0)
            
            frame_count = 0
            
            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect on every frame (or every Nth frame for performance)
                if frame_count % 8== 0:  # Change to 2 or 3 if too slow
                    helmet_detected, helmet_color, confidence, annotated_frame = detect_helmet_and_color(frame, model)
                    
                    # Show current frame
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame_rgb, use_container_width=True)
                    
                    # Show result for current frame
                    with result_placeholder_video.container():
                        show_result(helmet_detected, helmet_color, confidence)
                
                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
                frame_count += 1
            
            progress_bar.progress(1.0)
            st.success("✅ Video playback complete!")
            
            # # 🔹 FAST PROCESSING MODE (summary)
            #     progress_bar = st.progress(0)
            #     frame_placeholder = st.empty()
                
            #     frame_count = 0
            #     detections = []
                
            #     while cap.isOpened():
            #         ret, frame = cap.read()
            #         if not ret:
            #             break
                    
            #         # Process every 5th frame for speed
            #         if frame_count % 5 == 0:
            #             helmet_detected, helmet_color, confidence, annotated_frame = detect_helmet_and_color(frame, model)
            #             detections.append((helmet_detected, helmet_color, confidence))
                        
            #             # Show current frame
            #             annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            #             frame_placeholder.image(annotated_frame_rgb, use_container_width=True)
                        
            #             # Update progress
            #             progress = min(frame_count / total_frames, 1.0)
            #             progress_bar.progress(progress)
                    
            #         frame_count += 1
                
            progress_bar.progress(1.0)
                
            # # Summary
            # if detections:
            #     detected_count = sum(1 for d in detections if d[0])
            #     st.success(f"✅ Detection complete!")
                
            #     # Statistics
            #     col1, col2, col3 = st.columns(3)
            #     with col1:
            #         st.metric("Total Frames Processed", len(detections))
            #     with col2:
            #         st.metric("Helmets Detected", detected_count)
            #     with col3:
            #         st.metric("Detection Rate", f"{(detected_count/len(detections)*100):.1f}%")
                
            #     # Show most common color
            #     colors = [d[1] for d in detections if d[0]]
            #     if colors:
            #         color_counts = Counter(colors)
            #         st.info("🎨 **Detected Colors:**")
            #         for color, count in color_counts.most_common():
            #             st.write(f"   - **{color}**: {count} times ({count/len(colors)*100:.1f}%)")
            #     else:
            #         st.warning("⚠️ No helmets detected in this video")
        
            cap.release()
            
            # Cleanup temp file
            try:
                os.unlink(temp_video_path)
            except:
                pass
 
st.markdown("---")
st.markdown("**PETROCHOISE INTEGRATED SERVICES** | Helmet Safety Detection System")