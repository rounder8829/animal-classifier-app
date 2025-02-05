import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

# 구버전 API 사용으로 인한 출력되는 경고메시지만 제거
warnings.filterwarnings("ignore")

# 로컬 모델 경로
model_path = "model/saved_model"

# SavedModel 형식으로 모델을 로드 (tf.saved_model.load 사용)
@st.cache_resource
def load_model():
    return tf.saved_model.load(model_path)

model = load_model()

# 클래스 레이블 동적으로 로드
def load_class_names():
    # 모델의 출력 레이어 확인
    output_layer = model.layers[-1]
    num_classes = output_layer.output_shape[-1]
    
    # 실제 클래스 이름 확인 (예시)
    class_names = ['class_{}'.format(i) for i in range(num_classes)]
    
    # 클래스 이름 출력
    st.write(f"Number of classes: {num_classes}")
    st.write(f"Class names: {class_names}")
    
    return class_names


# 이미지 전처리 함수
def preprocess_image(img):
    img = img.resize((224, 224))  # PIL 이미지 크기 조정
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def main():
    fontRegistered()
    plt.rc('font', family='NanumGothic')

    st.title('동물인지 아닌지 예측하는 앱')

    # 클래스 레이블 동적으로 로드
    class_name = load_class_names()

    # 이미지 업로드
    upload_file = st.file_uploader('이미지 파일을 업로드 하세요.', type=['jpg', 'png', 'jpeg'])

    if upload_file is not None:
        # 이미지 표시
        img = Image.open(upload_file)
        st.image(img, caption='이미지 업로드', use_column_width=True)

        # 이미지 전처리
        img_array = preprocess_image(img)

        # 모델 정보 출력
        st.write(f"Model loaded: {model}")

        # 예측
        predictions = model(img_array)

        # 예측 결과 구조 확인
        st.write(f"Predictions structure: {predictions}")

        class_names = load_class_names()

        # 예측 결과 처리
        if isinstance(predictions, dict):
            predictions = predictions['predictions'].numpy()
        elif isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()
        
        predicted_class = np.argmax(predictions[0])
        predicted_confidence = np.max(predictions[0])

        # 예측 결과 출력
        if predicted_class < len(class_name):
            predicted_class_name = class_name[predicted_class]
            st.write(f"Predicted Class: {predicted_class_name}")
        else:
            st.write(f"Predicted Class: {predicted_class} (Class index out of range)")

        st.write(f"Predicted Confidence: {predicted_confidence:.2f}")

        # 추가 디버깅 정보
        st.write(f"Raw predictions: {predictions}")
        st.write(f"Predicted class index: {predicted_class}")

if __name__ == '__main__':
    main()
