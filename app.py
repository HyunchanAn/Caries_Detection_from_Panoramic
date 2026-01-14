import streamlit as st
from ultralytics import YOLO
from PIL import Image
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

model_path = "yolo11s.pt" # Default to small model (pretrained on COCO)
if model_source == "사용자 학습 모델":
    custom_model_path = st.sidebar.text_input("모델 경로 (.pt 파일)", "runs/detect/train/weights/best.pt")
    if os.path.exists(custom_model_path):
        model_path = custom_model_path
    else:
        st.sidebar.warning("지정된 경로에 모델이 없습니다. 기본 모델을 사용합니다.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

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
                
                # Plot results
                # results[0].plot() returns a BGR numpy array
                res_plotted = results[0].plot()
                res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR to RGB

            with col2:
                st.image(res_image, caption='분석 결과', use_container_width=True)
            
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
