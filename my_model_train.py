import tensorflow as tf
import numpy as np
import librosa
import glob
from sklearn.preprocessing import StandardScaler
import joblib

# 음성 특징 추출 함수 (MFCC만 사용)
def extract_mfcc(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 데이터 로드
user_files = glob.glob('received_audio/*.wav')  # 'me' 클래스 오디오 파일

# 특징 추출 및 스케일링
X_train = [extract_mfcc(file) for file in user_files]
X_train = np.array(X_train)

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler_me.pkl")  # 스케일러 저장

# CNN-LSTM Autoencoder 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((40, 1), input_shape=(40,)),  # 입력 형태 설정
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(40, activation='linear')  # 출력층, 원본과 같은 차원
])

model.compile(optimizer='adam', loss='mse')  # Autoencoder의 경우 MSE 손실 사용
model.fit(X_train, X_train, epochs=30, batch_size=16)
model.save("cnn_lstm_autoencoder_me.h5")
print("Autoencoder 모델이 성공적으로 저장되었습니다.")


# import numpy as np
# import librosa
# import glob
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import OneClassSVM
# import joblib

# # 음성 특징 추출 함수 (MFCC만 사용)
# def extract_mfcc(audio_file_path):
#     y, sr = librosa.load(audio_file_path, sr=16000)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     return np.mean(mfccs.T, axis=0)

# # 데이터 로드
# user_files = glob.glob('received_audio/*.wav')  # 'me' 클래스 오디오 파일

# # 특징 추출
# X_train = [extract_mfcc(file) for file in user_files]
# X_train = np.array(X_train)

# # 데이터 스케일링 (표준화)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# # One-Class SVM 모델 생성 및 학습
# model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)  # nu 값 조정 가능
# model.fit(X_train)

# # 모델과 스케일러 저장
# joblib.dump(model, "one_class_svm_model_me.pkl")
# joblib.dump(scaler, "scaler_me.pkl")
# print("One-Class SVM 모델이 성공적으로 저장되었습니다.")
