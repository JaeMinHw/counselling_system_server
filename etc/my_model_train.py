# 모델 학습하는 코드
import tensorflow as tf
import numpy as np
import librosa
import glob
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 데이터 증강 함수
def augment_audio(y, sr):
    """
    오디오 데이터에 증강 기법을 적용하여 데이터 다양화.
    - 노이즈 추가
    - 속도 변환
    - 피치 변환
    """
    noise = np.random.normal(0, 0.005, y.shape)  # 노이즈 추가
    y_noisy = y + noise

    y_stretched = librosa.effects.time_stretch(y, rate=1.2)  # 속도 증가
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)  # 피치 변환

    return [y_noisy, y_stretched, y_pitched]

# 음성 특징 추출 함수 (구간별 처리)
def extract_mfcc_segments(audio_file_path, segment_duration=1.0):
    """
    오디오 파일을 일정 길이의 구간으로 나누고, 각 구간에서 MFCC를 추출합니다.
    """
    y, sr = librosa.load(audio_file_path, sr=16000)
    segment_length = int(segment_duration * sr)
    segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

    features = []
    for seg in segments:
        if len(seg) == segment_length:
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40, n_fft = 1024)
            features.append(np.mean(mfcc.T, axis=0))
            # 증강 데이터를 추가
            for aug in augment_audio(seg, sr):
                if len(aug) >= segment_length:  # 길이 확인
                    aug_mfcc = librosa.feature.mfcc(y=aug[:segment_length], sr=sr, n_mfcc=40, n_fft = 1024)
                    features.append(np.mean(aug_mfcc.T, axis=0))

    return features

# 데이터 로드 및 처리
def load_and_process_data(data_path):
    """
    데이터 폴더에서 오디오 파일을 로드하고, MFCC를 추출하여 훈련 데이터를 만듭니다.
    """
    audio_files = glob.glob(os.path.join(data_path, '*.wav'))
    X = []
    for file in audio_files:
        print(f"Processing: {file}")
        X.extend(extract_mfcc_segments(file))  # 각 파일의 구간별 MFCC 추가
    return np.array(X)

# 데이터 준비
user_files_path = 'received_audio'  # 'me' 클래스 데이터 경로
X_train = load_and_process_data(user_files_path)

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler_me3.pkl")  # 스케일러 저장
print("스케일러가 저장되었습니다.")

# CNN-LSTM Autoencoder 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((40, 1), input_shape=(40,)),  # 입력 데이터 형태 변환
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),  # Dropout 추가로 과적합 방지
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(40, activation='linear')  # Autoencoder 복원 출력
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')
print("모델 학습을 시작합니다...")
model.fit(X_train, X_train, epochs=50, batch_size=8, validation_split=0.2)
model.save("cnn_lstm_autoencoder_me3.h5")
print("모델이 성공적으로 저장되었습니다.")


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
