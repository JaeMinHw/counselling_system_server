import tensorflow as tf
import numpy as np
import librosa
import joblib

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me.h5")
scaler = joblib.load("scaler_me.pkl")


threshold=0.2

# 음성 특징 추출 함수 (MFCC)
def extract_mfcc(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 입력 데이터를 준비하는 함수
def prepare_input(audio_file_path):
    features = extract_mfcc(audio_file_path)
    scaled_features = scaler.transform([features])  # 스케일링
    return scaled_features

# 모델 테스트 함수
def test_model(audio_file_path):
    # 입력 데이터 준비
    input_data = prepare_input(audio_file_path)

    # 모델 예측
    reconstructed = model.predict(input_data)

    # 재구성 손실 계산 (MSE)
    mse = np.mean((input_data - reconstructed) ** 2)

    # 임계값 기반 분류
    label = "me" if mse < threshold else "another"
    return label, mse

# 테스트 실행
test_audio_path = "C:/Users/User/Desktop/server/test/known1/jm (2).wav"  # 테스트할 오디오 파일 경로
label, mse = test_model(test_audio_path)
print(f"Predicted Label: {label}, MSE: {mse}")
