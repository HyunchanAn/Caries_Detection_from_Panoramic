import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

st.set_page_config(page_title="Caries Detection AI", layout="wide")

st.title("🦷 Panoramic Caries Detection (우식 탐지)")
st.markdown("""
이 어플리케이션은 **YOLOv11** 모델을 사용하여 파노라마 X-ray 이미지에서 치아 우식(Caries)을 탐지합니다.
""")

# Sidebar for Model Selection
st.sidebar.header("Model Settings")
model_source = st.sidebar.radio("모델 선택", ["기본 모델 (yolo11s.pt)", "사용자 학습 모델"])

model_path = "models/best.pt" # Default to newly trained model
if model_source == "사용자 학습 모델":
    custom_model_path = st.sidebar.text_input("모델 경로 (.pt 파일)", "models/best.pt")
    if os.path.exists(custom_model_path):
        model_path = custom_model_path
    else:
        st.sidebar.warning("지정된 경로에 모델이 없습니다. 기본 모델을 사용합니다.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
line_width = st.sidebar.slider("Line Width (선 굵기)", 1, 5, 2)
font_size = st.sidebar.slider("Font Size (글자 크기)", 5, 50, 15)

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

model = load_model(model_path)

# Main Interface
uploaded_file = st.file_uploader("파노라마 이미지 업로드", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='원본 이미지', use_container_width=True)

    if st.button("탐지 시작 (Detect)"):
        if model:
            with st.spinner("AI가 분석 중입니다..."):
                # Run inference
                results = model.predict(image, conf=conf_threshold)
                
                # Plot results: Draw ONLY boxes (labels=False)
                res_plotted = results[0].plot(line_width=line_width, labels=False)
                
                # Convert BGR (OpenCV) to RGB (PIL)
                res_image = Image.fromarray(res_plotted[..., ::-1])
                draw = ImageDraw.Draw(res_image, "RGBA") # Allow alpha
                
                # Try to load a Korean font
                try:
                    font_path = "C:/Windows/Fonts/malgun.ttf"
                    if not os.path.exists(font_path):
                        font_path = "arial.ttf"
                    
                    # Font for legend
                    font_size_legend = int(14 * (res_image.width / 1000))
                    font_legend = ImageFont.truetype(font_path, font_size_legend)
                    
                    # Font for confidence scores on teeth (Using Regular)
                    font_size_score = int(font_size * (res_image.width / 1000) * 0.8)
                    font_score = ImageFont.truetype(font_path, font_size_score)
                except:
                    font_legend = ImageFont.load_default()
                    font_score = ImageFont.load_default()

                scale = max(1, res_image.width // 1000)
                
                # --- Draw Confidence Scores with Background Colors ---
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    # Match color with class (consistent with legend)
                    class_colors = {
                        0: (0, 0, 255),    # Impacted (Blue)
                        1: (0, 255, 255),  # Caries (Cyan)
                        2: (255, 165, 0),  # Periapical (Orange)
                        3: (128, 0, 128)   # Deep Caries (Purple)
                    }
                    bg_color = class_colors.get(cls_id, (0, 0, 0))
                    
                    score_text = f"{conf:.2f}"
                    
                    # Calculate text background box size
                    bbox = draw.textbbox((xyxy[0], xyxy[1]), score_text, font=font_score)
                    # Draw background rectangle
                    draw.rectangle([bbox[0], xyxy[1] - font_size_score - 4, bbox[2] + 4, xyxy[1]], fill=bg_color)
                    # Draw score text (Conditional color: White for Impacted, Black for others)
                    text_color = (255, 255, 255) if cls_id == 0 else (0, 0, 0)
                    draw.text((xyxy[0] + 2, xyxy[1] - font_size_score - 2), score_text, font=font_score, fill=text_color)

                # --- Legend Overlay Start (Smaller & More Transparent) ---
                legend_items = [
                    {"name": "Impacted (매복치)", "color": (0, 0, 255)},
                    {"name": "Caries (충치)", "color": (0, 255, 255)},
                    {"name": "Periapical Lep. (치근단)", "color": (255, 165, 0)},
                    {"name": "Deep Caries (깊은충치)", "color": (128, 0, 128)}
                ]
                
                start_x, start_y = 15 * scale, 20 * scale
                spacing = 22 * scale # Reduced spacing
                
                # Draw HIGHER transparency background (alpha=80 instead of 128)
                # Calculate dynamic width based on text length
                max_text_width = 0
                for item in legend_items:
                    text_bbox = draw.textbbox((0, 0), item["name"], font=font_legend)
                    text_w = text_bbox[2] - text_bbox[0]
                    max_text_width = max(max_text_width, text_w)
                
                # Width = text_offset (20*scale) + max_text_width + padding (15*scale)
                bg_w = (20 * scale) + max_text_width + (15 * scale)
                bg_h = len(legend_items) * spacing + 10 * scale
                draw.rectangle([start_x - 8, start_y - 12, start_x + bg_w, start_y + bg_h - 10], fill=(0, 0, 0, 80))
                
                for i, item in enumerate(legend_items):
                    y_pos = start_y + (i * spacing)
                    # Draw small color box
                    draw.rectangle([start_x, y_pos - 6 * scale, start_x + 12 * scale, y_pos + 6 * scale], fill=item["color"])
                    # Draw text
                    draw.text((start_x + 20 * scale, y_pos - 10 * scale), item["name"], font=font_legend, fill=(255, 255, 255))
                # --- Legend Overlay End ---

            with col2:
                st.image(res_image, caption='분석 결과 (확률만 표시 & 범례 최적화)', use_container_width=True)
            
            # Show Detailed Results
            st.subheader("탐지된 객체 목록")
            boxes = results[0].boxes
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    st.write(f"- **{cls_name}**: {conf:.2%}")
            else:
                st.info("탐지된 객체가 없습니다.")
        else:
            st.error("모델이 로드되지 않았습니다.")

st.markdown("---")
st.markdown("Developed with YOLOv11 & Streamlit")
