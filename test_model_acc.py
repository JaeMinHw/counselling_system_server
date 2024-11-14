import numpy as np
import librosa
import tensorflow as tf
import joblib

# 학습한 모델과 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me.h5")
scaler = joblib.load("scaler_me.pkl")

# 음성 특징 추출 함수 (MFCC만 사용)
def extract_mfcc(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 음성 예측 함수
def predict_voice(audio_file_path, threshold=0.1):
    # 특징 추출 및 스케일링
    features = extract_mfcc(audio_file_path).reshape(1, -1)
    features = scaler.transform(features)

    # 모델로 복원된 데이터와 입력 데이터 간의 MSE 계산
    reconstructed = model.predict(features)
    mse = np.mean((features - reconstructed) ** 2)

    # 임계값을 사용해 결과 판정
    if mse < threshold:
        return "me", mse  # mse가 threshold보다 작으면 "me"로 판단
    else:
        return "another", mse  # mse가 threshold보다 크면 "another"로 판단

# 테스트 예측
test_file = 'C:/Users/User/Desktop/server/test/hjm4.wav'  # 테스트할 오디오 파일 경로
threshold = 4  # 임계값을 낮추거나 높여서 조정 가능
result, mse = predict_voice(test_file, threshold=threshold)
print(f"Prediction result for {test_file}: {result}")
print(f"MSE: {mse:.4f}")
print(f"Threshold used: {threshold}")



# import numpy as np
# import librosa
# import joblib

# # 모델과 스케일러 로드
# model = joblib.load("one_class_svm_model_me.pkl")
# scaler = joblib.load("scaler_me.pkl")

# # 음성 특징 추출 함수 (MFCC만 사용)
# def extract_mfcc(audio_file_path):
#     y, sr = librosa.load(audio_file_path, sr=16000)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     return np.mean(mfccs.T, axis=0)

# # 예측 함수 (One-Class SVM을 사용하여 "me" 또는 "another"로 분류)
# def predict_voice(audio_file_path):
#     # 특징 추출 및 스케일링
#     features = extract_mfcc(audio_file_path).reshape(1, -1)
#     features = scaler.transform(features)

#     # One-Class SVM 예측
#     prediction = model.predict(features)

#     # 1이면 "me", -1이면 "another"로 분류
#     return "me" if prediction >= 0.1 else "another"

# # 테스트 예측
# test_file = 'C:/Users/User/Desktop/server/test/jm2.wav'  # 테스트할 오디오 파일 경로
# result = predict_voice(test_file)
# print(f"Prediction result for {test_file}: {result}")
