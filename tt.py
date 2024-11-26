import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

# 모델 및 스케일러 로드
model = tf.keras.models.load_model("cnn_lstm_autoencoder_me3.h5")
scaler = joblib.load("scaler_me3.pkl")

threshold = 0.083

# 음성 특징 추출 함수 (MFCC)
def extract_mfcc(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024)
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

# 폴더 내 모든 파일 테스트 함수
def test_folder(folder_path, true_label):
    total_files = 0
    correct_predictions = 0
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):  # WAV 파일만 처리
            file_path = os.path.join(folder_path, file_name)
            predicted_label, mse = test_model(file_path)
            total_files += 1

            # 정확도 검사
            is_correct = (predicted_label == true_label)
            correct_predictions += int(is_correct)

            # 결과 저장
            results.append({
                "file": file_name,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "mse": mse,
                "correct": is_correct
            })

    # 정확도 계산
    accuracy = correct_predictions / total_files if total_files > 0 else 0
    return results, accuracy

# 테스트 실행
test_folder_path = "C:/Users/User/Desktop/server/test/known"  # 테스트할 폴더 경로
true_label = "me"  # 폴더의 실제 레이블
results, accuracy = test_folder(test_folder_path, true_label)

# 결과 출력
print(f"Accuracy: {accuracy * 100:.2f}%")
for result in results:
    print(f"File: {result['file']}, Predicted: {result['predicted_label']}, True: {result['true_label']}, MSE: {result['mse']:.5f}, Correct: {result['correct']}")
