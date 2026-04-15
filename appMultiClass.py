import base64
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
from collections import Counter

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
if 'mode' not in st.session_state:
    st.session_state.mode = None

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
        font-size: 18px;
        font-weight: bold;
        border: 3px solid white;
    }
    .detection-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #1a1a2e;
        border-left: 4px solid;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOGO AND HEADER
# ===============================
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
st.title("🪖 Multi-Object Helmet Detection System")
st.markdown("### Real-time detection of multiple helmets with color classification")

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
# MULTI-OBJECT DETECTION FUNCTION
# ===============================
def detect_all_helmets(frame, model, conf_threshold=0.5):
    """
    Detect ALL helmets in frame with their class names and confidence scores.
    Returns list of detections: [(class_name, confidence, box_coords), ...]
    """
    results = model(frame, conf=conf_threshold, verbose=False)
    
    detections = []
    annotated_frame = frame.copy()
    
    # Mapping colors for each class
    class_colors = {
        'helmet_blue': (255, 0, 0),       # Blue in BGR
        'helmet_white': (255, 255, 255),  # White
        'helmet_yellow': (0, 255, 255),   # Yellow in BGR
        'no_hard_hat': (0, 0, 255)        # Red for no helmet
    }
    
    for result in results:
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        # Process ALL detections, not just the best one
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
        
        for idx in range(len(boxes)):
            confidence = float(confidences[idx])
            class_id = int(class_ids[idx])
            box = xyxy_coords[idx]
            
            # Get class name from model
            helmet_class = model.names[class_id]
            
            # Skip "no_hard_hat" detections (optional - you can include them)
            if helmet_class == 'no_hard_hat':
                continue
            
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Store detection info
            detections.append({
                'class': helmet_class,
                'confidence': confidence,
                'box': (x1, y1, x2, y2),
                'color_name': helmet_class.replace('helmet_', '').upper()
            })
            
            # Get color for this class
            box_color = class_colors.get(helmet_class, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label with class name and confidence
            label = f"{helmet_class} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), box_color, -1)
            
            # Draw text
            text_color = (0, 0, 0) if helmet_class == 'helmet_white' else (255, 255, 255)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return detections, annotated_frame

# ===============================
# RESULT DISPLAY FUNCTION
# ===============================
def show_multi_results(detections):
    """Display all detected helmets in detail"""
    
    if not detections:
        st.markdown(
            f"""
            <div class="result-box" style="background-color:#CC3333; color:white;">
                ❌ NO HELMETS DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Summary box
    total_helmets = len(detections)
    st.markdown(
        f"""
        <div class="result-box" style="background-color:#00AA00; color:white;">
            ✅ <b>{total_helmets} HELMET(S) DETECTED</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Detailed detections
    st.markdown("### Detection Details:")
    
    # Create columns for each detection
    num_cols = min(3, total_helmets)  # Max 3 columns
    cols = st.columns(num_cols)
    
    for idx, detection in enumerate(detections):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            color_map = {
                'BLUE': '🔵',
                'WHITE': '⚪',
                'YELLOW': '🟡',
                'GREEN': '🟢'  
            }
            emoji = color_map.get(detection['color_name'], '🪖')
            
            st.metric(
                f"{emoji} {detection['color_name']}",
                f"{detection['confidence']:.1%}"
            )

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        # Try different paths
        model_paths = [
            "best multiclass.pt",
            "./best multiclass.pt",
            "best multiclass.pt",
            "./best multiclass.pt",
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = YOLO(path)
                return model, path
        
        # If none found, try loading directly
        model = YOLO("best.pt")
        return model, "best.pt"
        
    except Exception as e:
        raise e

try:
    model, model_path = load_model()
    st.success(f"✅ Model loaded from: {model_path}")
    st.info(f"📊 Model Classes: {', '.join(model.names.values())}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.error("⚠️ Make sure the model file is in the same directory as this script")
    st.stop()

# ===============================
# CONFIDENCE THRESHOLD SLIDER
# ===============================
st.sidebar.markdown("### ⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.95,
    value=0.5,
    step=0.05,
    help="Only show detections above this confidence level"
)

# ===============================
# MODE SELECTION
# ===============================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("📹 Real-Time Camera", use_container_width=True, key="camera_btn"):
        st.session_state.mode = "camera"

with col2:
    if st.button("📁 Upload Image / Video", use_container_width=True, key="upload_btn"):
        st.session_state.mode = "upload"

# ===============================
# CAMERA MODE - MULTI-OBJECT
# ===============================
if st.session_state.mode == "camera":
    st.info("🎥 Starting Camera Stream... Click 'Stop Camera' to end")
    
    stop_button = st.button("⏹️ Stop Camera", key="stop_camera")
    
    image_placeholder = st.empty()
    result_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Open camera
    cap = cv2.VideoCapture(1)
    
    if cap is None or not cap.isOpened():
        st.error("❌ Could not open camera. Try changing the camera index (0, 1, 2)")
    else:
        frame_count = 0
        detections_log = []
        max_helmets_frame = 0
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
            
            # Run multi-object detection
            detections, annotated_frame = detect_all_helmets(
                frame, model, conf_threshold=confidence_threshold
            )
            detections_log.append(detections)
            max_helmets_frame = max(max_helmets_frame, len(detections))
            
            # Convert BGR to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            image_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Show results
            with result_placeholder.container():
                show_multi_results(detections)
            
            frame_count += 1
        
        cap.release()
        
        # Show statistics
        if detections_log:
            with stats_placeholder.container():
                st.markdown("### 📊 Session Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Frames", len(detections_log))
                
                with col2:
                    frames_with_helmets = sum(1 for d in detections_log if d)
                    st.metric("Frames w/ Helmets", frames_with_helmets)
                
                with col3:
                    st.metric("Max Helmets/Frame", max_helmets_frame)
                
                with col4:
                    total_detections = sum(len(d) for d in detections_log)
                    st.metric("Total Detections", total_detections)
                
                # Helmet color distribution
                all_colors = []
                for frame_detections in detections_log:
                    for det in frame_detections:
                        all_colors.append(det['color_name'])
                
                if all_colors:
                    st.markdown("### 🎨 Helmet Color Distribution")
                    color_counts = Counter(all_colors)
                    
                    col1, col2, col3 = st.columns(3)
                    color_cols = [col1, col2, col3]
                    
                    for idx, (color, count) in enumerate(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)):
                        color_cols[idx % 3].metric(
                            f"🪖 {color}",
                            count,
                            "detections"
                        )
        
        st.info("✅ Camera stream stopped")

# ===============================
# UPLOAD MODE - MULTI-OBJECT
# ===============================
elif st.session_state.mode == "upload":
    file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if file:
        file_type = file.type
        
        image_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # ===== IMAGE MODE - MULTI-OBJECT =====
        if "image" in file_type:
            # Read image
            image = Image.open(file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Run multi-object detection
            detections, annotated_frame = detect_all_helmets(
                image_bgr, model, conf_threshold=confidence_threshold
            )
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            image_placeholder.image(annotated_frame_rgb, caption="Detection Result", use_container_width=True)
            
            with result_placeholder.container():
                show_multi_results(detections)
        
        # ===== VIDEO MODE - MULTI-OBJECT =====
        elif "video" in file_type:
            st.info("🎬 Processing video with multi-object detection...")
            
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(file.getbuffer())
                temp_video_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.write(f"📊 Video Info: {fps} FPS, {total_frames} frames")
            
            # UI elements
            stop_video = st.button("⏹️ Stop Video Processing", key="stop_video")
            frame_placeholder = st.empty()
            result_placeholder_video = st.empty()
            progress_bar = st.progress(0)
            
            frame_count = 0
            detections_log = []
            max_helmets_frame = 0
            
            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every frame with multi-object detection
                detections, annotated_frame = detect_all_helmets(
                    frame, model, conf_threshold=confidence_threshold
                )
                detections_log.append(detections)
                max_helmets_frame = max(max_helmets_frame, len(detections))
                
                # Show current frame
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, use_container_width=True)
                
                # Show results for current frame
                with result_placeholder_video.container():
                    show_multi_results(detections)
                
                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
                frame_count += 1
            
            progress_bar.progress(1.0)
            cap.release()
            
            # ===== VIDEO SUMMARY =====
            if detections_log:
                st.success("✅ Video processing complete!")
                
                st.markdown("### 📊 Video Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                frames_with_helmets = sum(1 for d in detections_log if d)
                total_detections = sum(len(d) for d in detections_log)
                
                with col1:
                    st.metric("Total Frames", len(detections_log))
                
                with col2:
                    st.metric("Frames w/ Helmets", frames_with_helmets)
                
                with col3:
                    st.metric("Max Helmets/Frame", max_helmets_frame)
                
                with col4:
                    st.metric("Total Detections", total_detections)
                
                # Average helmets per frame
                avg_helmets = total_detections / len(detections_log) if detections_log else 0
                st.metric("Avg Helmets/Frame", f"{avg_helmets:.2f}")
                
                # Helmet type distribution across entire video
                all_colors = []
                for frame_detections in detections_log:
                    for det in frame_detections:
                        all_colors.append(det['color_name'])
                
                if all_colors:
                    st.markdown("### 🎨 Helmet Type Distribution (Entire Video)")
                    color_counts = Counter(all_colors)
                    
                    col1, col2, col3 = st.columns(3)
                    color_cols = [col1, col2, col3]
                    
                    for idx, (color, count) in enumerate(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)):
                        percentage = (count / total_detections) * 100
                        color_cols[idx % 3].metric(
                            f"🪖 {color}",
                            count,
                            f"{percentage:.1f}% of all"
                        )
            else:
                st.warning("⚠️ No helmets detected in this video")
            
            # Cleanup
            try:
                os.unlink(temp_video_path)
            except:
                pass

st.markdown("---")
st.markdown("Made for **PETROCHOISE INTEGRATED SERVICES** | Multi-Object Helmet Safety Detection System")