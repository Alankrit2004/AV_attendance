import cv2
import numpy as np
import os
from db import connect_db

# def train_model():
#     USERS_DIR = "users"
#     MODEL_DIR = "trained_model"
#     MODEL_PATH = os.path.join(MODEL_DIR, "model.yml")
#     LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.txt")

#     os.makedirs(MODEL_DIR, exist_ok=True)
#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     faces = []
#     labels = []
#     label_map = {}

#     # ✅ Fetch username-label mapping from DB
#     conn = connect_db()
#     cursor = conn.cursor()
#     cursor.execute("SELECT username, label FROM users")
#     db_users = cursor.fetchall()
#     conn.close()

#     db_label_map = {username: label for username, label in db_users}

#     for username in os.listdir(USERS_DIR):
#         user_path = os.path.join(USERS_DIR, username)
#         if not os.path.isdir(user_path) or username not in db_label_map:
#             continue

#         label = db_label_map[username]
#         label_map[label] = username

#         for filename in os.listdir(user_path):
#             if filename.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png')):
#                 img_path = os.path.join(user_path, filename)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     faces.append(img)
#                     labels.append(label)

#     if not faces:
#         print("❌ No face images found. Please register users first.")
#         return False

#     print(f"✅ Training on {len(faces)} images...")
#     recognizer.train(faces, np.array(labels))
#     recognizer.save(MODEL_PATH)

#     with open(LABEL_MAP_PATH, "w") as f:
#         for label, name in label_map.items():
#             f.write(f"{label}:{name}\n")

#     print(f"✅ Model saved to {MODEL_PATH}")
#     print(f"✅ Label map saved to {LABEL_MAP_PATH}")
#     return True


def train_model():
    USERS_DIR = "users"
    MODEL_DIR = "trained_model"
    MODEL_PATH = os.path.join(MODEL_DIR, "model.yml")
    LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.txt")

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for folder_name in os.listdir(USERS_DIR):
        user_path = os.path.join(USERS_DIR, folder_name)
        if not os.path.isdir(user_path):
            continue

        # Assign readable label for sample datasets like s1, s2
        if folder_name.startswith("s") and folder_name[1:].isdigit():
            username = f"SampleUser_{folder_name[1:]}"
        else:
            username = folder_name

        label_map[current_label] = username

        for filename in os.listdir(user_path):
            if filename.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png')):
                img_path = os.path.join(user_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(current_label)

        current_label += 1

    if not faces:
        print("❌ No face images found. Please register users first.")
        return False

    print(f"✅ Training on {len(faces)} images...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, "w") as f:
        for label, name in label_map.items():
            f.write(f"{label}:{name}\n")

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Label map saved to {LABEL_MAP_PATH}")
    return True


# train_model()