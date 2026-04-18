# import base64
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import os
# import tempfile
# from collections import Counter

# # ===============================
# # PAGE CONFIG
# # ===============================
# st.set_page_config(
#     page_title="Helmet Detection System",
#     page_icon="🪖",
#     layout="wide"
# )

# # ===============================
# # SESSION STATE INITIALIZATION
# # ===============================
# if 'mode' not in st.session_state:
#     st.session_state.mode = None

# # ===============================
# # CUSTOM CSS
# # ===============================
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 10px;
#         height: 50px;
#         font-size: 18px;
#     }
#     .result-box {
#         padding: 20px;
#         border-radius: 10px;
#         text-align: center;
#         font-size: 18px;
#         font-weight: bold;
#         border: 3px solid white;
#     }
#     .detection-item {
#         padding: 10px;
#         margin: 5px 0;
#         border-radius: 5px;
#         background-color: #1a1a2e;
#         border-left: 4px solid;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ===============================
# # LOGO AND HEADER
# # ===============================
# try:
#     with open(r"E:\petroChoise\Task\Helmet-Detection-with-colors\images\logo.png", "rb") as f:
#         logo_data = base64.b64encode(f.read()).decode()
    
#     st.markdown(f"""
#         <style>
#         .header {{
#             display: flex;
#             align-items: center;
#             gap: 15px;
#             padding: 0px 0px;
#         }}
#         .header img {{
#             width: 100px;
#         }}
#         .company-name {{
#             font-size: 25px;
#             font-weight: bold;
#         }}
#         .subtitle {{
#             font-size: 20px;
#             color: white;
#         }}
#         </style>
 
#         <div class="header">
#             <img src="data:image/png;base64,{logo_data}">
#             <div>
#                 <div class="company-name">PETROCHOISE</div>
#                 <div class="subtitle">INTEGRATED SERVICES</div>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
# except FileNotFoundError:
#     st.markdown("""
#         <div style="text-align: center;">
#             <div style="font-size: 35px; font-weight: bold;">PETROCHOISE</div>
#             <div style="font-size: 20px; color: gray;">INTEGRATED SERVICES</div>
#         </div>
#     """, unsafe_allow_html=True)

# # ===============================
# # HEADER
# # ===============================
# st.title("🪖 Multi-Object Helmet Detection System")
# st.markdown("### Real-time detection of multiple helmets with color classification")

# # ===============================
# # BACKGROUND IMAGE
# # ===============================
# try:
#     with open(r"E:\petroChoise\Task\Helmet-Detection-with-colors\images\Photorealistic_high-quality_construction_area_insi-1776079121348.png", "rb") as f:
#         img_data = f.read()
#     b64_encoded = base64.b64encode(img_data).decode()
#     st.markdown(f"""
#         <style>
#         .stApp {{
#             background-image: url(data:image/png;base64,{b64_encoded});
#             background-size: cover;
#             background-attachment: fixed;
#         }}
#         </style>
#     """, unsafe_allow_html=True)
# except FileNotFoundError:
#     pass

# # ===============================
# # MULTI-OBJECT DETECTION FUNCTION
# # ===============================
# # def detect_all_helmets(frame, model, conf_threshold=0.5):
# #     """
# #     Detect ALL helmets in frame with their class names and confidence scores.
# #     Returns list of detections: [(class_name, confidence, box_coords), ...]
# #     """
# #     results = model(frame, conf=conf_threshold, verbose=False)
    
# #     detections = []
# #     annotated_frame = frame.copy()
    
# #     # Mapping colors for each class
# #     class_colors = {
# #         'helmet_blue': (255, 0, 0),       # Blue in BGR
# #         'helmet_white': (255, 255, 255),  # White
# #         'helmet_yellow': (0, 255, 255),   # Yellow in BGR
# #         'no_hard_hat': (0, 0, 255)        # Red for no helmet
# #     }
    
# #     for result in results:
# #         boxes = result.boxes
        
# #         if boxes is None or len(boxes) == 0:
# #             continue
        
# #         # Process ALL detections, not just the best one
# #         confidences = boxes.conf.cpu().numpy()
# #         class_ids = boxes.cls.cpu().numpy().astype(int)
# #         xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
        
# #         for idx in range(len(boxes)):
# #             confidence = float(confidences[idx])
# #             class_id = int(class_ids[idx])
# #             box = xyxy_coords[idx]
            
# #             # Get class name from model
# #             helmet_class = model.names[class_id]
            
# #             # Skip "no_hard_hat" detections (optional - you can include them)
# #             if helmet_class == 'no_hard_hat':
                
# #                 continue
            
# #             x1, y1, x2, y2 = box
            
# #             # Ensure coordinates are within bounds
# #             x1, y1 = max(0, x1), max(0, y1)
# #             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
# #             # Store detection info
# #             detections.append({
# #                 'class': helmet_class,
# #                 'confidence': confidence,
# #                 'box': (x1, y1, x2, y2),
# #                 'color_name': helmet_class.replace('helmet_', '').upper()
# #             })
            
# #             # Get color for this class
# #             box_color = class_colors.get(helmet_class, (0, 255, 0))
            
# #             # Draw bounding box
# #             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
# #             # Draw label with class name and confidence
# #             label = f"{helmet_class} {confidence:.2f}"
# #             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
# #             # Draw background for text
# #             cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
# #                           (x1 + label_size[0], y1), box_color, -1)
            
# #             # Draw text
# #             text_color = (0, 0, 0) if helmet_class == 'helmet_white' else (255, 255, 255)
# #             cv2.putText(annotated_frame, label, (x1, y1 - 5), 
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
# #     return detections, annotated_frame

# # # ===============================
# # # RESULT DISPLAY FUNCTION
# # # ===============================
# # def show_multi_results(detections):
# #     """Display all detected helmets in detail"""
    
# #     if not detections:
# #         st.markdown(
# #             f"""
# #             <div class="result-box" style="background-color:#CC3333; color:white;">
# #                 ❌ NO HELMETS DETECTED
# #             </div>
# #             """,
# #             unsafe_allow_html=True
# #         )
# #         return
    
# #     # Summary box
# #     total_helmets = len(detections)
# #     st.markdown(
# #         f"""
# #         <div class="result-box" style="background-color:#00AA00; color:white;">
# #             ✅ <b>{total_helmets} HELMET(S) DETECTED</b>
# #         </div>
# #         """,
# #         unsafe_allow_html=True
# #     )
    
# #     # Detailed detections
# #     st.markdown("### Detection Details:")
    
# #     # Create columns for each detection
# #     num_cols = min(3, total_helmets)  # Max 3 columns
# #     cols = st.columns(num_cols)
    
# #     for idx, detection in enumerate(detections):
# #         col_idx = idx % num_cols
        
# #         with cols[col_idx]:
# #             color_map = {
# #                 'BLUE': '🔵',
# #                 'WHITE': '⚪',
# #                 'YELLOW': '🟡',
# #                 'GREEN': '🟢'  
# #             }
# #             emoji = color_map.get(detection['color_name'], '🪖')
            
# #             st.metric(
# #                 f"{emoji} {detection['color_name']}",
# #                 f"{detection['confidence']:.1%}"
# #             )
# def detect_all_helmets(frame, model, conf_threshold=0.5):
#     """
#     Detect ALL helmets in frame with their class names and confidence scores.
    
#     Model Classes: 
#     - helmet_blue
#     - helmet_white
#     - helmet_yellow
#     - NO-Helmet
    
#     Returns list of detections with class info
#     """
#     results = model(frame, conf=conf_threshold, verbose=False)
    
#     detections = []
#     annotated_frame = frame.copy()
    
#     # Mapping colors for each class (BGR format for OpenCV)
#     class_colors = {
#         'helmet_blue': (255, 0, 0),       # Blue in BGR
#         'helmet_white': (255, 255, 255),  # White
#         'helmet_yellow': (0, 255, 255),   # Yellow in BGR
#         'NO-Helmet': (0, 0, 255)          # Red for NO-Helmet
#     }
    
#     for result in results:
#         boxes = result.boxes
        
#         if boxes is None or len(boxes) == 0:
#             continue
        
#         # Process ALL detections, not just the best one
#         confidences = boxes.conf.cpu().numpy()
#         class_ids = boxes.cls.cpu().numpy().astype(int)
#         xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
        
#         for idx in range(len(boxes)):
#             confidence = float(confidences[idx])
#             class_id = int(class_ids[idx])
#             box = xyxy_coords[idx]
            
#             # Get class name from model (exact name from model.names)
#             helmet_class = model.names[class_id]
            
#             # Skip "NO-Helmet" detections (background/no helmet detected)
#             if helmet_class == 'NO-Helmet':
#                 continue
            
#             x1, y1, x2, y2 = box
            
#             # Ensure coordinates are within bounds
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
#             # Extract color name for display (e.g., "BLUE" from "helmet_blue")
#             if helmet_class.startswith('helmet_'):
#                 color_name = helmet_class.replace('helmet_', '').upper()
#             else:
#                 color_name = helmet_class.upper()
            
#             # Store detection info
#             detections.append({
#                 'class': helmet_class,
#                 'confidence': confidence,
#                 'box': (x1, y1, x2, y2),
#                 'color_name': color_name
#             })
            
#             # Get color for this class
#             box_color = class_colors.get(helmet_class, (0, 255, 0))
            
#             # Draw bounding box
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
#             # Draw label with class name and confidence
#             label = f"{color_name} {confidence:.2f}"
#             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
#             # Draw background for text
#             cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
#                           (x1 + label_size[0], y1), box_color, -1)
            
#             # Draw text (black for white helmet, white for others)
#             text_color = (0, 0, 0) if helmet_class == 'helmet_white' else (255, 255, 255)
#             cv2.putText(annotated_frame, label, (x1, y1 - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
#     return detections, annotated_frame
 
# # ===============================
# # RESULT DISPLAY FUNCTION
# # ===============================
# def show_multi_results(detections):
#     """
#     Display all detected helmets with their classes and confidence scores.
    
#     Model Classes:
#     - helmet_blue → 🔵 BLUE
#     - helmet_white → ⚪ WHITE
#     - helmet_yellow → 🟡 YELLOW
#     """
    
#     if not detections:
#         st.markdown(
#             f"""
#             <div class="result-box" style="background-color:#CC3333; color:white;">
#                 ❌ NO HELMETS DETECTED
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#         return
    
#     # Summary box
#     total_helmets = len(detections)
#     st.markdown(
#         f"""
#         <div class="result-box" style="background-color:#00AA00; color:white;">
#             ✅ <b>{total_helmets} HELMET(S) DETECTED</b>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
#     # Detailed detections
#     st.markdown("### 🎯 Detection Details:")
    
#     # Create columns for each detection
#     num_cols = min(3, total_helmets)  # Max 3 columns
#     cols = st.columns(num_cols)
    
#     # Color emoji mapping based on model classes
#     color_map = {
#         'BLUE': '🔵',
#         'WHITE': '⚪',
#         'YELLOW': '🟡'
#     }
    
#     for idx, detection in enumerate(detections):
#         col_idx = idx % num_cols
        
#         with cols[col_idx]:
#             # Get emoji for this color
#             emoji = color_map.get(detection['color_name'], '🪖')
            
#             # Display metric with class name and confidence
#             st.metric(
#                 f"{emoji} {detection['color_name']}",
#                 f"{detection['confidence']:.1%}",
#                 delta=f"Class: {detection['class']}"
#             )
# # ===============================
# # LOAD MODEL
# # ===============================
# @st.cache_resource
# def load_model():
#     try:
#         # Try different paths
#         model_paths = [
#             "bestMULTI.pt",
#             "./bestMULTI.pt",
#             "bestMULTI.pt",
#             "./bestMULTI.pt",
#         ]
        
#         for path in model_paths:
#             if os.path.exists(path):
#                 model = YOLO(path)
#                 return model, path
        
#         # If none found, try loading directly
#         model = YOLO("bestMULTI.pt")
#         return model, "best.pt"
        
#     except Exception as e:
#         raise e

# try:
#     model, model_path = load_model()
#     st.success(f"✅ Model loaded from: {model_path}")
#     st.info(f"📊 Model Classes: {', '.join(model.names.values())}")
# except Exception as e:
#     st.error(f"❌ Error loading model: {e}")
#     st.error("⚠️ Make sure the model file is in the same directory as this script")
#     st.stop()

# # ===============================
# # CONFIDENCE THRESHOLD SLIDER
# # ===============================
# # st.sidebar.markdown("### ⚙️ Detection Settings")
# # confidence_threshold = st.sidebar.slider(
# #     "Confidence Threshold",
# #     min_value=0.1,
# #     max_value=0.95,
# #     value=0.5,
# #     step=0.05,
# #     help="Only show detections above this confidence level"
# # )

# # ===============================
# # MODE SELECTION
# # ===============================
# st.markdown("---")
# col1, col2 = st.columns(2)

# with col1:
#     if st.button("📹 Real-Time Camera", use_container_width=True, key="camera_btn"):
#         st.session_state.mode = "camera"

# with col2:
#     if st.button("📁 Upload Image / Video", use_container_width=True, key="upload_btn"):
#         st.session_state.mode = "upload"

# # ===============================
# # CAMERA MODE - MULTI-OBJECT
# # ===============================
# if st.session_state.mode == "camera":
#     st.info("🎥 Starting Camera Stream... Click 'Stop Camera' to end")
    
#     stop_button = st.button("⏹️ Stop Camera", key="stop_camera")
    
#     image_placeholder = st.empty()
#     result_placeholder = st.empty()
#     stats_placeholder = st.empty()
    
#     # Open camera
#     cap = cv2.VideoCapture(1)
    
#     if cap is None or not cap.isOpened():
#         st.error("❌ Could not open camera. Try changing the camera index (0, 1, 2)")
#     else:
#         frame_count = 0
#         detections_log = []
#         max_helmets_frame = 0
        
#         while cap.isOpened() and not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to grab frame from camera")
#                 break
            
#             # Run multi-object detection
#             detections, annotated_frame = detect_all_helmets(
#                 frame, model, conf_threshold=0.5
#             )
#             detections_log.append(detections)
#             max_helmets_frame = max(max_helmets_frame, len(detections))
            
#             # Convert BGR to RGB for display
#             annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
#             # Display frame
#             image_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
#             # Show results
#             with result_placeholder.container():
#                 show_multi_results(detections)
            
#             frame_count += 1
        
#         cap.release()
        
#         # Show statistics
#         if detections_log:
#             with stats_placeholder.container():
#                 st.markdown("### 📊 Session Statistics")
                
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     st.metric("Total Frames", len(detections_log))
                
#                 with col2:
#                     frames_with_helmets = sum(1 for d in detections_log if d)
#                     st.metric("Frames w/ Helmets", frames_with_helmets)
                
#                 with col3:
#                     st.metric("Max Helmets/Frame", max_helmets_frame)
                
#                 with col4:
#                     total_detections = sum(len(d) for d in detections_log)
#                     st.metric("Total Detections", total_detections)
                
#                 # Helmet color distribution
#                 all_colors = []
#                 for frame_detections in detections_log:
#                     for det in frame_detections:
#                         all_colors.append(det['color_name'])
                
#                 if all_colors:
#                     st.markdown("### 🎨 Helmet Color Distribution")
#                     color_counts = Counter(all_colors)
                    
#                     col1, col2, col3 = st.columns(3)
#                     color_cols = [col1, col2, col3]
                    
#                     for idx, (color, count) in enumerate(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)):
#                         color_cols[idx % 3].metric(
#                             f"🪖 {color}",
#                             count,
#                             "detections"
#                         )
        
#         st.info("✅ Camera stream stopped")

# # ===============================
# # UPLOAD MODE - MULTI-OBJECT
# # ===============================
# elif st.session_state.mode == "upload":
#     file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
#     if file:
#         file_type = file.type
        
#         image_placeholder = st.empty()
#         result_placeholder = st.empty()
        
#         # ===== IMAGE MODE - MULTI-OBJECT =====
#         if "image" in file_type:
#             # Read image
#             image = Image.open(file)
#             image_np = np.array(image)
            
#             # Convert RGB to BGR for OpenCV
#             if len(image_np.shape) == 3 and image_np.shape[2] == 3:
#                 image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#             else:
#                 image_bgr = image_np
            
#             # Run multi-object detection
#             detections, annotated_frame = detect_all_helmets(
#                 image_bgr, model, conf_threshold=0.5
#             )
            
#             # Convert back to RGB for display
#             annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
#             # Display
#             image_placeholder.image(annotated_frame_rgb, caption="Detection Result", use_container_width=True)
            
#             with result_placeholder.container():
#                 show_multi_results(detections)
        
#         # ===== VIDEO MODE - MULTI-OBJECT =====
#         elif "video" in file_type:
#             st.info("🎬 Processing video with multi-object detection...")
            
#             # Save uploaded video temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#                 tmp_file.write(file.getbuffer())
#                 temp_video_path = tmp_file.name
            
#             # Open video
#             cap = cv2.VideoCapture(temp_video_path)
            
#             # Video properties
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             st.write(f"📊 Video Info: {fps} FPS, {total_frames} frames")
            
#             # UI elements
#             stop_video = st.button("⏹️ Stop Video Processing", key="stop_video")
#             frame_placeholder = st.empty()
#             result_placeholder_video = st.empty()
#             progress_bar = st.progress(0)
            
#             frame_count = 0
#             detections_log = []
#             max_helmets_frame = 0
            
#             while cap.isOpened() and not stop_video:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Process every frame with multi-object detection
#                 detections, annotated_frame = detect_all_helmets(
#                     frame, model, conf_threshold=0.5
#                 )
#                 detections_log.append(detections)
#                 max_helmets_frame = max(max_helmets_frame, len(detections))
                
#                 # Show current frame
#                 annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                 frame_placeholder.image(annotated_frame_rgb, use_container_width=True)
                
#                 # Show results for current frame
#                 with result_placeholder_video.container():
#                     show_multi_results(detections)
                
#                 # Update progress
#                 progress = min(frame_count / total_frames, 1.0)
#                 progress_bar.progress(progress)
                
#                 frame_count += 1
            
#             progress_bar.progress(1.0)
#             cap.release()
            
#             # ===== VIDEO SUMMARY =====
#             if detections_log:
#                 st.success("✅ Video processing complete!")
                
#                 st.markdown("### 📊 Video Analysis Summary")
                
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 frames_with_helmets = sum(1 for d in detections_log if d)
#                 total_detections = sum(len(d) for d in detections_log)
                
#                 with col1:
#                     st.metric("Total Frames", len(detections_log))
                
#                 with col2:
#                     st.metric("Frames w/ Helmets", frames_with_helmets)
                
#                 with col3:
#                     st.metric("Max Helmets/Frame", max_helmets_frame)
                
#                 with col4:
#                     st.metric("Total Detections", total_detections)
                
#                 # Average helmets per frame
#                 avg_helmets = total_detections / len(detections_log) if detections_log else 0
#                 st.metric("Avg Helmets/Frame", f"{avg_helmets:.2f}")
                
#                 # Helmet type distribution across entire video
#                 all_colors = []
#                 for frame_detections in detections_log:
#                     for det in frame_detections:
#                         all_colors.append(det['color_name'])
                
#                 if all_colors:
#                     st.markdown("### 🎨 Helmet Type Distribution (Entire Video)")
#                     color_counts = Counter(all_colors)
                    
#                     col1, col2, col3 = st.columns(3)
#                     color_cols = [col1, col2, col3]
                    
#                     for idx, (color, count) in enumerate(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)):
#                         percentage = (count / total_detections) * 100
#                         color_cols[idx % 3].metric(
#                             f"🪖 {color}",
#                             count,
#                             f"{percentage:.1f}% of all"
#                         )
#             else:
#                 st.warning("⚠️ No helmets detected in this video")
            
#             # Cleanup
#             try:
#                 os.unlink(temp_video_path)
#             except:
#                 pass

# st.markdown("---")
# ____________________________________________________________________________
# ____________________________________________________________________________
# ____________________________________________________________________________
# ____________________________________________________________________________
# ____________________________________________________________________________



# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import tempfile

# # --- 1. إعدادات الصفحة و Session State ---
# st.set_page_config(
#     page_title="Petrochoice HSE Monitor",
#     page_icon="🛢️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # تعريف الـ Session State لحفظ السجل والـ IDs
# if 'violators_log' not in st.session_state:
#     st.session_state.violators_log = []
# if 'logged_violator_ids' not in st.session_state:
#     st.session_state.logged_violator_ids = set()

# # --- 2. إعداد الخلفية ---
# page_bg_img = '''
# <style>
# .stApp {
#     background-image: url("https://images.unsplash.com/photo-1587313632749-3e405a8167f1?q=80&w=1920&auto=format&fit=crop");
#     background-size: cover;
#     background-position: center;
#     background-attachment: fixed;
# }
# html, body, [class*="css"] { font-weight: bold; }
# .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
#     color: white;
# }
# </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)

# # --- 3. تحميل المودل ---
# @st.cache_resource
# def load_model():
#     model = YOLO("lastl.pt") 
#     return model

# model = load_model()

# # --- 4. الهيدر (Header) ---
# st.markdown('<div style="background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
# col_logo, col_title = st.columns([1, 5])
# with col_logo:
#     try:
#         logo = Image.open("logo.png")
#         st.image(logo, use_container_width=True) 
#     except FileNotFoundError:
#         st.warning("اللوجو غير موجود.")
# with col_title:
#     st.title("Petrochoice Safety & Detection System")
#     st.markdown("### 🛡️ نظام مراقبة الفيديو الذكي للسلامة والصحة المهنية (HSE)")
# st.markdown('</div><br>', unsafe_allow_html=True)

# # --- 5. القائمة الجانبية (Sidebar) واختيار المصدر ---
# st.sidebar.markdown('### ⚙️ إعدادات النظام')
# # 🟢 إضافة اختيار مصدر المراقبة
# source = st.sidebar.radio("اختر مصدر المراقبة:", ("فيديو مسجل (Upload)", "كاميرا مباشرة (Webcam)"))

# st.sidebar.markdown("---")
# st.sidebar.markdown('### 🗑️ إدارة السجل')
# if st.sidebar.button("مسح صور المخالفين المحفوظة"):
#     st.session_state.violators_log = []
#     st.session_state.logged_violator_ids = set()
#     st.sidebar.success("تم مسح السجل بنجاح!")

# # إعداد متغير الـ Camera/Video Capture
# cap = None

# # 🟢 معالجة اختيار المصدر
# if source == "فيديو مسجل (Upload)":
#     video_file_buffer = st.sidebar.file_uploader("قم برفع فيديو من الموقع لفحصه", type=['mp4', 'avi', 'mov'])
#     if video_file_buffer is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False) 
#         tfile.write(video_file_buffer.read())
#         cap = cv2.VideoCapture(tfile.name)
# else:
#     st.sidebar.info("سيتم استخدام الكاميرا المتصلة بالجهاز.")
#     if st.sidebar.button("🎥 تشغيل الكاميرا"):
#         # رقم 0 بيشغل الكاميرا الأساسية، لو عندك كاميرات تانية جرب 1 أو 2
#         cap = cv2.VideoCapture(0) 

# # --- 6. معالجة الفيديو / الكاميرا وعرض النتائج ---
# if cap is not None:
#     col_video, col_log = st.columns([2.5, 1])
    
#     with col_video:
#         st.markdown('<div style="background-color: rgba(0, 0, 0, 0.7); padding: 15px; border-radius: 10px;">', unsafe_allow_html=True)
#         st.markdown("### 🎥 المراقبة المباشرة:")
#         stframe = st.empty()
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         stop_button = st.button("🛑 إيقاف المراقبة")
        
#     with col_log:
#         st.markdown('<div style="background-color: rgba(0, 0, 0, 0.7); padding: 15px; border-radius: 10px;">', unsafe_allow_html=True)
#         st.markdown("### 🗂️ سجل المخالفين المحفوظ:")
#         log_placeholder = st.empty()
#         st.markdown('</div>', unsafe_allow_html=True)

#     last_log_count = -1

#     while cap.isOpened() and not stop_button:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.resize(frame, (640, 480))
        
#         # استخدام Tracker لتتبع الأشخاص
#         results = model.track(frame, persist=True, conf=0.4, verbose=False)
        
#         res_plotted = results[0].plot()
#         res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
#         stframe.image(res_plotted_rgb, channels="RGB", use_container_width=True)
        
#         # استخراج المخالفين بناءً على الـ ID الفريد
#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#             track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
#             for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
#                 class_name = model.names[cls_id]
                
#                 if class_name.lower() in ['not_helmet', 'no_helmet', 'not helmet', 'no helmet', 'without_helmet']:
                    
#                     if track_id not in st.session_state.logged_violator_ids:
#                         x1, y1, x2, y2 = box
#                         h, w = frame.shape[:2]
#                         x1, y1 = max(0, x1), max(0, y1)
#                         x2, y2 = min(w, x2), min(h, y2)
                        
#                         crop_bgr = frame[y1:y2, x1:x2]
#                         if crop_bgr.size > 0:
#                             crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            
#                             st.session_state.violators_log.append(crop_rgb)
#                             st.session_state.logged_violator_ids.add(track_id)
        
#         # تحديث شاشة السجل
#         if len(st.session_state.violators_log) != last_log_count:
#             with log_placeholder.container():
#                 if len(st.session_state.violators_log) == 0:
#                     st.success("السجل نظيف. لا يوجد مخالفات.")
#                 else:
#                     for idx, img in enumerate(reversed(st.session_state.violators_log)):
#                         st.image(img, caption="غير مصرح له", use_container_width=True)
            
#             last_log_count = len(st.session_state.violators_log)
            
#     cap.release()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import base64
from pathlib import Path
import time

# --- Page Config ---
st.set_page_config(
    page_title="PetroChoice HSE Monitor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
if 'violators_log' not in st.session_state:
    st.session_state.violators_log = []
if 'logged_violator_ids' not in st.session_state:
    st.session_state.logged_violator_ids = set()
if 'total_persons' not in st.session_state:
    st.session_state.total_persons = 0
if 'compliant' not in st.session_state:
    st.session_state.compliant = 0
if 'violations_count' not in st.session_state:
    st.session_state.violations_count = 0

# --- Helper: image to base64 ---
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# --- Background & Full CSS ---
def inject_css(bg_b64, logo_b64):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;500;600&display=swap');

    /* ===== GLOBAL RESET ===== */
    html, body, [class*="css"] {{
        font-family: 'Exo 2', sans-serif !important;
    }}

    /* ===== BACKGROUND ===== */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Dark overlay on top of background */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: linear-gradient(
            135deg,
            rgba(0, 10, 30, 0.82) 0%,
            rgba(0, 20, 50, 0.75) 50%,
            rgba(10, 5, 20, 0.85) 100%
        );
        z-index: 0;
        pointer-events: none;
    }}

    /* ===== MAIN CONTENT LAYER ===== */
    .main .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }}

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {{
        background: rgba(0, 15, 40, 0.92) !important;
        border-right: 1px solid rgba(255, 160, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }}
    [data-testid="stSidebar"] * {{
        color: #e8eaf0 !important;
    }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: #FFA500 !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 2px;
        font-size: 13px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,165,0,0.3);
        padding-bottom: 6px;
        margin-bottom: 12px;
    }}

    /* Sidebar radio */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: #c8cdd8 !important;
        font-size: 14px !important;
    }}

    /* ===== HEADER CARD ===== */
    .petro-header {{
        background: linear-gradient(135deg, rgba(0,20,60,0.95) 0%, rgba(10,30,70,0.90) 100%);
        border: 1px solid rgba(255,160,0,0.4);
        border-top: 3px solid #FFA500;
        border-radius: 12px;
        padding: 18px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }}

    .petro-logo-wrap {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid rgba(255,160,0,0.5);
        overflow: hidden;
        flex-shrink: 0;
        background: rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .petro-logo-wrap img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    .petro-title-block {{
        flex: 1;
    }}
    .petro-company {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #FFA500;
        letter-spacing: 4px;
        text-transform: uppercase;
        line-height: 1;
        margin: 0;
        text-shadow: 0 0 20px rgba(255,165,0,0.3);
    }}
    .petro-subtitle {{
        font-family: 'Exo 2', sans-serif;
        font-size: 13px;
        color: #8899bb;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin: 4px 0 0 0;
    }}
    .petro-system {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #c8daf0;
        letter-spacing: 1px;
        margin-top: 8px;
    }}

    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,200,100,0.15);
        border: 1px solid rgba(0,200,100,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        color: #00e676;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }}
    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00e676;
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.4; transform: scale(0.8); }}
    }}

    /* ===== METRIC CARDS ===== */
    .metric-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }}
    .metric-card {{
        background: rgba(0, 15, 40, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }}
    .metric-card.orange::before {{ background: #FFA500; }}
    .metric-card.green::before {{ background: #00e676; }}
    .metric-card.red::before {{ background: #ff4444; }}
    .metric-card.blue::before {{ background: #4488ff; }}

    .metric-label {{
        font-size: 10px;
        color: #556688;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Rajdhani', sans-serif;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 32px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 2px;
    }}
    .metric-card.orange .metric-value {{ color: #FFA500; }}
    .metric-card.green .metric-value {{ color: #00e676; }}
    .metric-card.red .metric-value {{ color: #ff4444; }}
    .metric-card.blue .metric-value {{ color: #4488ff; }}
    .metric-sub {{
        font-size: 11px;
        color: #445566;
        font-family: 'Exo 2', sans-serif;
    }}

    /* ===== VIDEO PANEL ===== */
    .panel-card {{
        background: rgba(0, 12, 35, 0.90);
        border: 1px solid rgba(255,160,0,0.2);
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }}
    .panel-header {{
        background: rgba(0,0,0,0.4);
        border-bottom: 1px solid rgba(255,160,0,0.2);
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .panel-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: #FFA500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }}
    .panel-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #FFA500;
        animation: pulse 2s infinite;
    }}
    .panel-body {{
        padding: 14px;
    }}

    /* ===== LOG PANEL ===== */
    .log-panel {{
        background: rgba(0, 12, 35, 0.90);
        border: 1px solid rgba(255,60,60,0.25);
        border-radius: 12px;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }}
    .log-header {{
        background: rgba(100,0,0,0.3);
        border-bottom: 1px solid rgba(255,60,60,0.2);
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .log-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: #ff4444;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }}
    .log-body {{
        padding: 12px;
        max-height: 500px;
        overflow-y: auto;
    }}
    .violator-entry {{
        background: rgba(255,40,40,0.08);
        border: 1px solid rgba(255,60,60,0.2);
        border-left: 3px solid #ff4444;
        border-radius: 6px;
        padding: 8px;
        margin-bottom: 10px;
    }}
    .violator-label {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 11px;
        color: #ff6666;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}
    .clean-log {{
        text-align: center;
        padding: 30px 10px;
        color: #00e676;
        font-family: 'Rajdhani', sans-serif;
        font-size: 13px;
        letter-spacing: 2px;
    }}

    /* ===== ALERT BANNER ===== */
    .alert-banner {{
        background: rgba(255,40,0,0.15);
        border: 1px solid rgba(255,60,0,0.5);
        border-left: 4px solid #ff3300;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 10px 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 14px;
        color: #ff6644;
        letter-spacing: 1px;
        animation: alertFlash 1s infinite;
    }}
    @keyframes alertFlash {{
        0%, 100% {{ border-left-color: #ff3300; }}
        50% {{ border-left-color: rgba(255,51,0,0.2); }}
    }}

    /* ===== STREAMLIT OVERRIDES ===== */
    .stButton > button {{
        background: rgba(255,160,0,0.1) !important;
        border: 1px solid rgba(255,160,0,0.5) !important;
        color: #FFA500 !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 2px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        background: rgba(255,160,0,0.25) !important;
        border-color: #FFA500 !important;
    }}

    [data-testid="stFileUploader"] {{
        background: rgba(0,20,60,0.5) !important;
        border: 1px dashed rgba(255,160,0,0.3) !important;
        border-radius: 8px !important;
    }}

    /* Hide streamlit default elements */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stDecoration"] {{ display: none; }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{ width: 4px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255,160,0,0.4); border-radius: 2px; }}

    /* Image display fix */
    .stImage img {{
        border-radius: 6px !important;
    }}

    /* Column spacing */
    [data-testid="column"] {{
        padding: 0 6px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# =============================================
#               MAIN APP
# =============================================

bg_b64 = img_to_base64("logo1.png")  # Use logo file as bg source
logo_b64 = img_to_base64("logo.png")

inject_css(bg_b64, logo_b64)

# ===== HEADER =====
logo_html = f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else '🛢️'

st.markdown(f"""
<div class="petro-header">
    <div class="petro-logo-wrap">{logo_html}</div>
    <div class="petro-title-block">
        <p class="petro-company">PetroChoice</p>
        <p class="petro-subtitle">Integrated Services</p>
        <p class="petro-system">HSE Intelligent Safety Monitoring System</p>
    </div>
    <div style="text-align:right; display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
        <div class="status-badge">
            <div class="status-dot"></div>
            System Active
        </div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:11px; color:#445566; letter-spacing:1px;">
            AI-POWERED · REAL-TIME · HSE COMPLIANT
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== METRICS =====
total_p = st.session_state.total_persons
compliant = st.session_state.compliant
violations = st.session_state.violations_count
compliance_rate = int((compliant / total_p * 100) if total_p > 0 else 100)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card blue">
        <div class="metric-label">Persons Detected</div>
        <div class="metric-value">{total_p}</div>
        <div class="metric-sub">Total in frame</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Compliant</div>
        <div class="metric-value">{compliant}</div>
        <div class="metric-sub">Helmet worn</div>
    </div>
    <div class="metric-card red">
        <div class="metric-label">Violations</div>
        <div class="metric-value">{violations}</div>
        <div class="metric-sub">No helmet detected</div>
    </div>
    <div class="metric-card orange">
        <div class="metric-label">Compliance Rate</div>
        <div class="metric-value">{compliance_rate}%</div>
        <div class="metric-sub">Site safety score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.markdown("### ⚙️ SYSTEM SETTINGS")
source = st.sidebar.radio("Monitoring Source:", ("Uploaded Video", "Live Camera"))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 DETECTION CONFIG")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗑️ LOG MANAGEMENT")
if st.sidebar.button("🗑️ Clear Violation Log"):
    st.session_state.violators_log = []
    st.session_state.logged_violator_ids = set()
    st.session_state.total_persons = 0
    st.session_state.compliant = 0
    st.session_state.violations_count = 0
    st.sidebar.success("Log cleared successfully.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'Rajdhani',sans-serif; font-size:10px; color:#334455; letter-spacing:1px; line-height:1.8;">
    <div>PETROCHOICE HSE MONITOR</div>
    <div>VERSION 2.0 · AI ENGINE</div>
    <div>YOLO DETECTION MODEL</div>
    <div style="margin-top:8px; color:#FFA500;">● AUTHORIZED PERSONNEL ONLY</div>
</div>
""", unsafe_allow_html=True)

# ===== MODEL LOAD =====
@st.cache_resource
def load_model():
    return YOLO("last (3)m.pt")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.sidebar.error("Model 'lastl.pt' not found.")

# ===== INPUT SOURCE =====
cap = None
if source == "Uploaded Video":
    video_file = st.sidebar.file_uploader("Upload Site Video", type=['mp4', 'avi', 'mov'])
    if video_file and model_loaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
else:
    if st.sidebar.button("🎥 START CAMERA"):
        cap = cv2.VideoCapture(1)

# ===== MAIN DISPLAY =====
if cap is not None and model_loaded:
    col_vid, col_log = st.columns([2.5, 1])

    with col_vid:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-dot"></div>
                <span class="panel-title">Live Monitoring Feed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        stframe = st.empty()
        alert_placeholder = st.empty()
        stop_button = st.button("⬛ STOP MONITORING")

    with col_log:
        st.markdown("""
        <div class="log-panel">
            <div class="log-header">
                <span style="color:#ff4444; font-size:14px;">⚠</span>
                <span class="log-title">Violation Log</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        log_placeholder = st.empty()

    last_log_count = -1
    frame_count = 0

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        if frame_count % 8== 0:  # Change to 2 or 3 if too slow

            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

            # Count classes in frame
            frame_persons = 0
            frame_helmets = 0
            frame_violations = 0

        if results[0].boxes is not None:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for cls_id in class_ids:
                class_name = model.names[cls_id].lower()
                if 'person' in class_name:
                    frame_persons += 1
                elif 'helmet' in class_name and 'not' not in class_name and 'no_' not in class_name:
                    frame_helmets += 1
                elif any(x in class_name for x in ['not_helmet','No-Helmet','not helmet','no helmet','without_helmet']):
                    frame_violations += 1

        st.session_state.total_persons = frame_persons
        st.session_state.compliant = frame_helmets
        st.session_state.violations_count = frame_violations + len(st.session_state.violators_log)

        # Draw frame
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        stframe.image(res_rgb, channels="RGB", use_container_width=True)

        # Alert banner
        if frame_violations > 0:
            alert_placeholder.markdown(f"""
            <div class="alert-banner">
                ⚠ ALERT — {frame_violations} PERSON(S) WITHOUT HELMET DETECTED — UNAUTHORIZED ACCESS
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

        # Log violators
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                class_name = model.names[cls_id].lower()
                if class_name.lower() in ['No-Helmet', 'no_helmet', 'not helmet', 'no-helmet', 'no helmet']:
                    if track_id not in st.session_state.logged_violator_ids:
                        x1, y1, x2, y2 = box
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.session_state.violators_log.append({
                                'img': crop_rgb,
                                'id': track_id,
                                'time': time.strftime("%H:%M:%S")
                            })
                            st.session_state.logged_violator_ids.add(track_id)

        # Update log
        if len(st.session_state.violators_log) != last_log_count:
            with log_placeholder.container():
                if len(st.session_state.violators_log) == 0:
                    st.markdown('<div class="clean-log">✓ NO VIOLATIONS LOGGED</div>', unsafe_allow_html=True)
                else:
                    for entry in reversed(st.session_state.violators_log):
                        img = entry['img'] if isinstance(entry, dict) else entry
                        t = entry.get('time', '--:--:--') if isinstance(entry, dict) else ''
                        tid = entry.get('id', '?') if isinstance(entry, dict) else '?'
                        st.markdown(f'<div class="violator-label">⚠ ID #{tid} · {t} · UNAUTHORIZED</div>', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
            last_log_count = len(st.session_state.violators_log)

    cap.release()

else:
    # Empty state
    st.markdown("""
    <div class="panel-card" style="text-align:center; padding:60px 20px;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:48px; color:rgba(255,160,0,0.2); margin-bottom:16px;">⬤</div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:16px; color:#445566; letter-spacing:3px; text-transform:uppercase;">
            Awaiting Video Source
        </div>
        <div style="font-family:'Exo 2',sans-serif; font-size:13px; color:#334455; margin-top:8px;">
            Upload a video or activate live camera to begin HSE monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)
