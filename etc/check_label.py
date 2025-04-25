# import os

# CACHE_DIR = r"C:/Users/User/Desktop/cursor_model_make/01.데이터/cache/features"
# LABEL_ROOT = r"C:/Users/User/Desktop/cursor_model_make/01.데이터/1.Training/라벨링데이터/TL1/1.감정"
# LOG_PATH = os.path.join(CACHE_DIR, "rename_log.txt")

# # 1. JSON 기준으로 매핑 테이블 구축
# def build_json_to_emotion_map():
#     mapping = {}
#     for emotion_folder in os.listdir(LABEL_ROOT):
#         if '.' not in emotion_folder:
#             continue
#         emotion_id = emotion_folder.split('.')[0]
#         emotion_path = os.path.join(LABEL_ROOT, emotion_folder)
#         for speaker_folder in os.listdir(emotion_path):
#             speaker_path = os.path.join(emotion_path, speaker_folder)
#             for file in os.listdir(speaker_path):
#                 if file.endswith(".json"):
#                     json_name = file.replace(".json", "")
#                     mapping[json_name] = emotion_id
#     return mapping

# # 2. npy 이름에 json 이름이 포함되어 있는지 확인 후 rename
# def rename_npy_files_partial_match():
#     mapping = build_json_to_emotion_map()
#     renamed = 0
#     log_lines = []
#     unmatched = []

#     for fname in os.listdir(CACHE_DIR):
#         if not fname.endswith(".npy"):
#             continue
#         fpath = os.path.join(CACHE_DIR, fname)

#         match = None
#         for json_name in mapping:
#             if json_name in fname:
#                 match = json_name
#                 break

#         if match:
#             emotion_id = mapping[match]
#             new_name = f"{emotion_id}_{match}.npy"
#             new_path = os.path.join(CACHE_DIR, new_name)
#             os.rename(fpath, new_path)
#             renamed += 1
#             log_lines.append(f"{fname} → {new_name}")
#         else:
#             unmatched.append(fname)

#     with open(LOG_PATH, "w", encoding="utf-8") as f:
#         f.write("\n".join(log_lines))

#     print(f"✅ 변경된 파일 수: {renamed}")
#     print(f"❌ 매핑 실패 파일 수: {len(unmatched)}")
#     if unmatched:
#         print("⚠️ 매핑 실패 파일 예시:")
#         print("\n".join(unmatched[:5]))

# rename_npy_files_partial_match()


import os

CACHE_DIR = r"C:/Users/User/Desktop/cursor_model_make/01.데이터/cache/features"
DRY_RUN = False  # ✅ True일 때는 실제 파일 이름은 바꾸지 않고 로그만 출력

def fix_0001_0001_files(dry_run=True):
    fixed = 0
    conflicts = 0
    i=0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".npy"):
            continue
        if "0001_0001" in fname:
            new_name = fname.replace("0001_0001", "0001")
            old_path = os.path.join(CACHE_DIR, fname)
            new_path = os.path.join(CACHE_DIR, new_name)

            if os.path.exists(new_path):
                print(f"⚠️ 이름 충돌 (이미 존재): {new_name}")
                conflicts += 1
                continue

            if dry_run:
                print(f"🔎 [예상 변경] {fname} → {new_name}")
            else:
                os.rename(old_path, new_path)
                print(f"✅ 변경됨: {fname} → {new_name}")
                fixed += 1

    print("\n🔁 총 예상 변경 수:", fixed if not dry_run else "(dry run이므로 변경 없음)")
    if conflicts > 0:
        print(f"🚨 이름 충돌로 인해 건너뛴 파일 수: {conflicts}")

# 실행
fix_0001_0001_files(dry_run=DRY_RUN)
