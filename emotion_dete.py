import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ✅ 모델과 테스트 파일 경로
MODEL_PATH = "1_best_model.h5"
TEST_WAV = "C:/Users/User/Desktop/server/t2.wav"  # 예측할 .wav 파일 경로

# ✅ 감정 라벨 정의 (모델 학습 시 사용했던 순서와 일치하도록)
EMOTIONS = {
    '1': '기쁨',
    '2': '슬픔',
    '3': '분노',
    '4': '불안',
    '5': '상처',
    '6': '당황',
    '7': '중립'
}

# ✅ EMOTIONS를 리스트로 정렬해서 모델 출력 인덱스와 매핑
# 키 정렬 결과: ['1', '2', ..., '7'] → ['기쁨', '슬픔', ..., '중립']
EMOTION_LABELS = [EMOTIONS[k] for k in sorted(EMOTIONS.keys())]

# ✅ 특징 추출 함수 (json 없이)
def extract_features_for_test(wav_path, max_pad_len=300):
    try:
        audio, sample_rate = librosa.load(wav_path, sr=None, res_type='kaiser_fast')

        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)), 'constant')

        # 특성 추출
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)

        # 특성 병합 및 정규화
        feature_list = [mfccs, chroma, mel, contrast, tonnetz, zcr, spectral_centroid, spectral_rolloff, spectral_bandwidth]
        min_frames = min(f.shape[1] for f in feature_list)
        feature_list = [f[:, :min_frames] for f in feature_list]
        features = np.vstack(feature_list).T

        # 패딩 또는 자르기
        if features.shape[0] < max_pad_len:
            features = np.pad(features, ((0, max_pad_len - features.shape[0]), (0, 0)), mode='constant')
        else:
            features = features[:max_pad_len, :]

        return features
    except Exception as e:
        print(f"❌ 특징 추출 중 오류 발생: {str(e)}")
        return None

# ✅ 감정 예측 함수
def predict_emotion(wav_path):
    model = load_model(MODEL_PATH)
    features = extract_features_for_test(wav_path)
    if features is None:
        print("❌ 특징 추출 실패. 예측 중단.")
        return

    # 모델 입력 형태로 변환
    features = np.expand_dims(features, axis=0)

    # 예측
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    
    # 예측된 감정
    if predicted_index < len(EMOTION_LABELS):
        predicted_emotion = EMOTION_LABELS[predicted_index]
    else:
        predicted_emotion = "알 수 없음"

    print("🎧 예측된 감정:", predicted_emotion)
    print("📊 확률 분포:")
    for i, prob in enumerate(prediction[0]):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"라벨{i}"
        print(f"  {label}: {prob:.4f}")

# ✅ 실행
predict_emotion(TEST_WAV)
