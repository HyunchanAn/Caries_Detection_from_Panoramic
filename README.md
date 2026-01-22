# Caries Detection from Panoramic Radiographs
(파노라마 방사선 사진을 이용한 치아 우식 탐지)

## 프로젝트 개요 (Project Overview)
이 프로젝트는 치과 파노라마 X-ray 이미지에서 우식(Caries)을 자동으로 탐지하는 딥러닝 모델을 개발하고, 이를 임상에서 쉽게 테스트해볼 수 있도록 Streamlit 기반의 웹 UI를 제공하는 것을 목표로 합니다. **특히, 성인뿐만 아니라 혼합치열기(소아) 환자의 데이터까지 학습하여 다양한 연령대에서 활용 가능하도록 개선되었습니다.**

## 주요 기능 (Features)
- **Object Detection**: YOLOv11 모델을 사용하여 다음 4가지 핵심 병소를 탐지:
    - `Caries` (충치)
    - `Deep Caries` (심한 충치 / 치수염)
    - `Periapical Lesion` (치근단 병소)
    - `Impacted` (매복치 / 발육 이상)
- **Streamlit UI**: 
    - 웹 브라우저를 통해 손쉽게 이미지를 업로드하고 분석 결과 확인.
    - 분석 결과 이미지에 영문/한글 라벨 범례 자동 표시.
- **Data Support**: DENTEX 및 Pediatric 데이터셋 통합 지원.

## 설치 및 실행 (Installation & Usage)

### 1. 환경 설정 (Environment Setup)
Python 3.8 이상이 필요합니다.
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 준비 (Dataset Preparation)
기본 데이터셋은 `data/` 폴더에 위치합니다. 
- DENTEX 데이터셋 변환: `src/data_converter.py`
- 소아 데이터셋 추가: `src/convert_pediatric.py` (자동으로 통합됨)

### 3. 모델 학습 (Training)
```bash
python train.py
```
학습된 모델은 `runs/detect/`와 `models/` 폴더에 저장됩니다.

### 4. 웹 애플리케이션 실행 (Run Web App)
로컬에서 앱을 실행하여 테스트할 수 있습니다.
```bash
python -m streamlit run app.py
```

## 배포 (Deployment)
이 프로젝트는 **Streamlit Cloud**에 최적화되어 있습니다.
1. GitHub 저장소에 Push.
2. Streamlit Cloud에서 해당 저장소 연결.
3. `app.py`를 메인 파일로 설정하여 배포.

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
