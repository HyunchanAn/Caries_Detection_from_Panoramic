# Caries Detection from Panoramic Radiographs
(파노라마 방사선 사진을 이용한 치아 우식 탐지)

## 프로젝트 개요 (Project Overview)
이 프로젝트는 치과 파노라마 X-ray 이미지에서 우식(Caries)을 자동으로 탐지하는 딥러닝 모델을 개발하고, 이를 임상에서 쉽게 테스트해볼 수 있도록 Streamlit 기반의 웹 UI를 제공하는 것을 목표로 합니다.

## 주요 기능 (Features)
- **Object Detection**: YOLOv11 모델을 사용하여 우식 부위를 Bounding Box로 탐지
- **Streamlit UI**: 웹 브라우저를 통해 손쉽게 이미지를 업로드하고 분석 결과 확인
- **Data Support**: DENTEX Challenge 2023 데이터셋 구조 지원 (변환 스크립트 포함)

## 설치 및 실행 (Installation & Usage)

### 1. 환경 설정 (Environment Setup)
Python 3.8 이상이 필요합니다.
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 준비 (Dataset Preparation)
DENTEX Challenge 2023 데이터셋이나 기타 파노라마 데이터셋을 `data/` 폴더에 위치시킵니다.
DENTEX 데이터셋의 경우 `src/data_converter.py`를 사용하여 YOLO 포맷으로 변환해야 합니다.
- python src/download_data.py

### 3. 모델 학습 (Training)
```bash
python train.py
```

### 4. 웹 애플리케이션 실행 (Run Web App)
```bash
streamlit run app.py
```

## 기술 스택 (Tech Stack)
- **Model**: YOLOv11 (Ultralytics)
- **Framework**: PyTorch
- **UI**: Streamlit
- **Image Processing**: OpenCV, Pillow

## 참고 문헌 및 레퍼런스 (References)
1. **DENTEX Challenge 2023**: [Link](https://dentex.grand-challenge.org/)
2. **YOLOv11**: [Ultralytics](https://github.com/ultralytics/ultralytics)
3. **Detection of Dental Anomalies in Digital Panoramic Images Using YOLO**: Uğur Şevik et al. (2025)

## 라이선스 (License)
MIT License
