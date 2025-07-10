from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

from deepface import DeepFace

from db import register_user_to_db, register_user_ref_to_db
from db import fetch_all_ref_users_with_embeddings, fetch_all_users_with_embeddings
from db import log_attendance_ref, log_attendance_user
from db import update_user_ref_in_db
from scipy.spatial.distance import cosine

from threading import Lock




app = Flask(__name__)
CORS(app)

# Constants
USERS_DIR = "users"
MODEL_PATH = "trained_model/model.yml"
LABEL_MAP_PATH = "trained_model/label_map.txt"
CONFIDENCE_THRESHOLD = 80  # Adjust as needed

# Create users and model directories
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs("trained_model", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global variables to avoid reloading in every request
recognizer = None
label_map = {}

#Global embedding chaches and lock
users_normal_cache = []
users_normal_embeddings = None

users_ref_cache = []
users_ref_embeddings = None
cache_lock = Lock()

def load_embedding_caches():
    global users_normal_cache, users_ref_cache, users_normal_embeddings, users_ref_embeddings
    with cache_lock:
        users_normal_cache = fetch_all_users_with_embeddings()
        users_ref_cache = fetch_all_ref_users_with_embeddings()

        users_normal_embeddings = np.array([u['embedding'] for u in users_normal_cache]) if users_normal_cache else np.empty((0, 512))
        users_ref_embeddings = np.array([u['embedding'] for u in users_ref_cache]) if users_ref_cache else np.empty((0, 512))


@app.route('/register_face', methods=['POST'])
def register_user():
    print("Register endpoint hit")
    try:
        username = request.form.get("username")
        email = request.form.get("email")
        mobilenumber = request.form.get("mobilenumber")

        if not username:
            return jsonify({
                "isSuccess": False,
                "message": "Username is required"
            }), 400

        if not email:
            return jsonify({
                "isSuccess": False,
                "message": "Email is required"
            }), 400

        if not mobilenumber:
            return jsonify({
                "isSuccess": False,
                "message": "Mobile number is required"
            }), 400

        if 'image' not in request.files:
            return jsonify({
                "isSuccess": False,
                "message": "No image file uploaded"
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "isSuccess": False,
                "message": "Empty file name"
            }), 400

        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({
                "isSuccess": False,
                "message": "Invalid image"
            }), 400

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({
                "isSuccess": False,
                "message": "Invalid image"
            }), 400

        # Resize image to 224x224
        img = cv2.resize(img, (224, 224))
        # Preprocess
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Face detection
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            return jsonify({
                "isSuccess": False,
                "message": "No face detected"
            }), 400

        # Save image
        user_folder = os.path.join(USERS_DIR, username)
        os.makedirs(user_folder, exist_ok=True)
        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)

        # Try to register the user (will raise Exception if user already exists)
        try:
            register_user_to_db(
                username=username,
                email=email,
                mobile=mobilenumber,
                image_path=save_path
            )
            load_embedding_caches()
        except Exception as e:
            return jsonify({
                "isSuccess": False,
                "message": str(e)
            }), 400

        return jsonify({
            "isSuccess": True,
            "message": f"User '{username}' registered successfully",
            "imagePath": save_path
        })
        

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "isSuccess": False,
            "message": f"Server error: {str(e)}"
        }), 500


@app.route('/register_face_v2', methods=['POST'])
def register_v2():
    print("Register V2 endpoint hit")
    try:
        id_val = int(request.form.get("id", 0))  # Default to 0 if not provided
        ref_id = request.form.get("ref_id")
        ref_type = request.form.get("ref_type")

        if not ref_id:
            return jsonify({
                "isSuccess": False,
                "message": "ref_id is required"
            }), 400

        if not ref_type:
            return jsonify({
                "isSuccess": False,
                "message": "ref_type is required"
            }), 400

        if 'image' not in request.files:
            return jsonify({
                "isSuccess": False,
                "message": "No image file uploaded"
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "isSuccess": False,
                "message": "Empty file name"
            }), 400

        # Decode image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({
                "isSuccess": False,
                "message": "Invalid image"
            }), 400

        # Resize image to 224x224
        img = cv2.resize(img, (224, 224))

        # Preprocessing
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Face detection
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            return jsonify({
                "isSuccess": False,
                "message": "No face detected"
            }), 400

        # Save image
        safe_folder = f"{ref_type}_{ref_id}".replace(" ", "_")
        user_folder = os.path.join(USERS_DIR, safe_folder)
        os.makedirs(user_folder, exist_ok=True)
        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)

        try:
            if id_val == 0:
                # Register new user in users_ref
                register_user_ref_to_db(ref_id, ref_type, save_path)
                message = f"User '{ref_id}' registered successfully"
            elif id_val == 1:
                # Update existing user in users_ref
                # You may want to add more fields as needed
                update_user_ref_in_db(ref_id, ref_type, save_path)
                message = f"User '{ref_id}' updated successfully"
            else:
                return jsonify({
                    "isSuccess": False,
                    "message": "Invalid id value"
                }), 400

            load_embedding_caches()
        except Exception as reg_err:
            return jsonify({
                "isSuccess": False,
                "message": str(reg_err)
            }), 400

        return jsonify({
            "isSuccess": True,
            "message": message,
            "imagePath": save_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "isSuccess": False,
            "message": f"Server error: {str(e)}"
        }), 500


# @app.route('/register_face_v2', methods=['POST'])
# def register_v2():
#     print("Register V2 endpoint hit")
#     try:
#         ref_id = request.form.get("ref_id")
#         ref_type = request.form.get("ref_type")

#         if not ref_id:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "ref_id is required"
#             }), 400

#         if not ref_type:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "ref_type is required"
#             }), 400

#         if 'image' not in request.files:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "No image file uploaded"
#             }), 400

#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "Empty file name"
#             }), 400

#         # Decode image
#         img_array = np.frombuffer(file.read(), np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is None:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "Invalid image"
#             }), 400

#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is None:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "Invalid image"
#             }), 400

#         # Resize image to 224x224
#         img = cv2.resize(img, (224, 224))

#         # Preprocessing
#         img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
#         img = cv2.GaussianBlur(img, (3, 3), 0)

#         # Face detection
#         faces = face_cascade.detectMultiScale(
#             cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
#             scaleFactor=1.1,
#             minNeighbors=5
#         )
#         if len(faces) == 0:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": "No face detected"
#             }), 400

#         # Save image
#         safe_folder = f"{ref_type}_{ref_id}".replace(" ", "_")
#         user_folder = os.path.join(USERS_DIR, safe_folder)
#         os.makedirs(user_folder, exist_ok=True)
#         save_path = os.path.join(user_folder, "1.jpg")
#         cv2.imwrite(save_path, img)

#         # Register in users_ref table
#         try:
#             register_user_ref_to_db(ref_id, ref_type, save_path)
#             load_embedding_caches()
#         except Exception as reg_err:
#             return jsonify({
#                 "isSuccess": False,
#                 "message": str(reg_err)
#             }), 400

#         return jsonify({
#             "isSuccess": True,
#             "message": f"User '{ref_id}' registered successfully",
#             "imagePath": save_path
#         })
#         load_embedding_caches()

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "isSuccess": False,
#             "message": f"Server error: {str(e)}"
#         }), 500





# @app.route('/recognize_face', methods=['POST'])
# def recognize():
#     if 'image' not in request.files:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No image uploaded"
#         }), 400

#     file = request.files['image']
#     img_array = np.frombuffer(file.read(), np.uint8)
#     input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if input_img is None:
#         return jsonify({
#             "isSuccess": False,
#             "message": "Invalid image"
#         }), 400

#     # Preprocessing
#     input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
#     input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

#     # Face detection
#     faces = face_cascade.detectMultiScale(
#         cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
#         scaleFactor=1.1,
#         minNeighbors=5
#     )
#     if len(faces) == 0:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No face detected"
#         }), 400

#     try:
#         input_embedding = DeepFace.represent(
#             img_path=input_img, model_name="SFace", enforce_detection=False
#         )[0]['embedding']
#     except Exception as e:
#         return jsonify({
#             "isSuccess": False,
#             "message": f"Embedding failed: {str(e)}"
#         }), 500

#     # Fetch embeddings from both tables
#     with cache_lock:
#         users_normal = users_normal_cache.copy()  # From `users` table
#         users_ref = users_ref_cache.copy()  # From `users_ref` table

#     best_user = None
#     best_distance = float("inf")
#     source = None  # 'users' or 'users_ref'

#     # Compare with users table
#     for user in users_normal:
#         distance = cosine(input_embedding, np.array(user['embedding']))
#         if distance < best_distance:
#             best_user = user
#             best_distance = distance
#             source = 'users'

#     # Compare with users_ref table
#     for user in users_ref:
#         distance = cosine(input_embedding, np.array(user['embedding']))
#         if distance < best_distance:
#             best_user = user
#             best_distance = distance
#             source = 'users_ref'

#     THRESHOLD = 0.4
#     if best_distance < THRESHOLD and best_user:
#         status = None

#         if source == 'users':
#             status = log_attendance_user(best_user['user_id'])
#             identity = {
#                 "username": best_user.get("username"),
#                 "email": best_user.get("email"),
#                 "mobile": best_user.get("mobile")
#             }
#         elif source == 'users_ref':
#             status = log_attendance_ref(best_user['ref_id'])
#             identity = {
#                 "ref_id": best_user.get("ref_id"),
#                 "ref_type": best_user.get("ref_type")
#             }
#         else:
#             identity = {"user_id": best_user.get("user_id")}

#         return jsonify({
#             "isSuccess": True,
#             "message": f"User recognized. Attendance: {status}",
#             "identity": identity,
#             "status": status
#         })

#     else:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No matching user found"
#         }), 404

@app.route('/recognize_face', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({
            "isSuccess": False,
            "message": "No image uploaded"
        }), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if input_img is None:
        return jsonify({
            "isSuccess": False,
            "message": "Invalid image"
        }), 400

    # Preprocessing
    input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if input_img is None:
        return jsonify({
            "isSuccess": False,
            "message": "Invalid image"
        }), 400

    # Resize image to 224x224
    input_img = cv2.resize(input_img, (224, 224))
    input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
    input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

    # Face detection
    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5
    )
    if len(faces) == 0:
        return jsonify({
            "isSuccess": False,
            "message": "No face detected"
        }), 400

    try:
        input_embedding = DeepFace.represent(
            img_path=input_img, model_name="SFace", enforce_detection=False
        )[0]['embedding']
    except Exception as e:
        return jsonify({
            "isSuccess": False,
            "message": f"Embedding failed: {str(e)}"
        }), 500

    # Vectorized similarity search
    with cache_lock:
        users_normal = users_normal_cache
        users_ref = users_ref_cache
        normal_embs = users_normal_embeddings
        ref_embs = users_ref_embeddings

    best_user = None
    best_distance = float("inf")
    source = None

    def vectorized_cosine_distances(embeddings, input_embedding):
        if embeddings.shape[0] == 0:
            return np.array([])
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        inp_norm = input_embedding / np.linalg.norm(input_embedding)
        return 1 - np.dot(emb_norm, inp_norm)

    # Compare with users table (vectorized)
    if normal_embs is not None and normal_embs.shape[0] > 0:
        distances = vectorized_cosine_distances(normal_embs, np.array(input_embedding))
        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_user = users_normal[min_idx]
            best_distance = distances[min_idx]
            source = 'users'

    # Compare with users_ref table (vectorized)
    if ref_embs is not None and ref_embs.shape[0] > 0:
        distances = vectorized_cosine_distances(ref_embs, np.array(input_embedding))
        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_user = users_ref[min_idx]
            best_distance = distances[min_idx]
            source = 'users_ref'

    THRESHOLD = 0.4
    if best_distance < THRESHOLD and best_user:
        status = None
        if source == 'users':
            status = log_attendance_user(best_user['user_id'])
            identity = {
                "username": best_user.get("username"),
                "email": best_user.get("email"),
                "mobile": best_user.get("mobile")
            }
        elif source == 'users_ref':
            status = log_attendance_ref(best_user['ref_id'])
            identity = {
                "ref_id": best_user.get("ref_id"),
                "ref_type": best_user.get("ref_type")
            }
        else:
            identity = {"user_id": best_user.get("user_id")}

        return jsonify({
            "isSuccess": True,
            "message": f"User recognized. Attendance: {status}",
            "identity": identity,
            "status": status
        })

    else:
        return jsonify({
            "isSuccess": False,
            "message": "No matching user found"
        }), 404


@app.route('/recognize_face_v2', methods=['POST'])
def recognize_v2():
    if 'image' not in request.files:
        return jsonify({
            "isSuccess": False,
            "message": "No image uploaded"
        }), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if input_img is None:
        return jsonify({
            "isSuccess": False,
            "message": "Invalid image"
        }), 400

    # Preprocessing
    input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if input_img is None:
        return jsonify({
            "isSuccess": False,
            "message": "Invalid image"
        }), 400

    # Resize image to 224x224
    input_img = cv2.resize(input_img, (224, 224))
    input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
    input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

    # Face detection
    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5
    )
    if len(faces) == 0:
        return jsonify({
            "isSuccess": False,
            "message": "No face detected"
        }), 400

    try:
        input_embedding = DeepFace.represent(
            img_path=input_img, model_name="SFace", enforce_detection=False
        )[0]['embedding']
    except Exception as e:
        return jsonify({
            "isSuccess": False,
            "message": f"Embedding failed: {str(e)}"
        }), 500

    # Vectorized similarity search
    with cache_lock:
        users_normal = users_normal_cache
        users_ref = users_ref_cache
        normal_embs = users_normal_embeddings
        ref_embs = users_ref_embeddings

    best_user = None
    best_distance = float("inf")
    source = None

    def vectorized_cosine_distances(embeddings, input_embedding):
        if embeddings.shape[0] == 0:
            return np.array([])
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        inp_norm = input_embedding / np.linalg.norm(input_embedding)
        return 1 - np.dot(emb_norm, inp_norm)

    # Compare with users table (vectorized)
    if normal_embs is not None and normal_embs.shape[0] > 0:
        distances = vectorized_cosine_distances(normal_embs, np.array(input_embedding))
        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_user = users_normal[min_idx]
            best_distance = distances[min_idx]
            source = 'users'

    # Compare with users_ref table (vectorized)
    if ref_embs is not None and ref_embs.shape[0] > 0:
        distances = vectorized_cosine_distances(ref_embs, np.array(input_embedding))
        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_user = users_ref[min_idx]
            best_distance = distances[min_idx]
            source = 'users_ref'

    THRESHOLD = 0.4
    if best_distance < THRESHOLD and best_user:
        if source == 'users':
            identity = {
                "username": best_user.get("username"),
                "email": best_user.get("email"),
                "mobile": best_user.get("mobile")
            }
        elif source == 'users_ref':
            identity = {
                "ref_id": best_user.get("ref_id"),
                "ref_type": best_user.get("ref_type")
            }
        else:
            identity = {"user_id": best_user.get("user_id")}

        return jsonify({
            "isSuccess": True,
            "message": "User matched",
            "identity": identity
        })

    else:
        return jsonify({
            "isSuccess": False,
            "message": "No matching user found"
        }), 404



# @app.route('/recognize_face_v2', methods=['POST'])
# def recognize_v2():
#     if 'image' not in request.files:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No image uploaded"
#         }), 400

#     file = request.files['image']
#     img_array = np.frombuffer(file.read(), np.uint8)
#     input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     if input_img is None:
#         return jsonify({
#             "isSuccess": False,
#             "message": "Invalid image"
#         }), 400

#     # Preprocessing
#     input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
#     input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

#     # Face detection
#     faces = face_cascade.detectMultiScale(
#         cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
#         scaleFactor=1.1,
#         minNeighbors=5
#     )
#     if len(faces) == 0:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No face detected"
#         }), 400

#     try:
#         input_embedding = DeepFace.represent(
#             img_path=input_img, model_name="SFace", enforce_detection=False
#         )[0]['embedding']
#     except Exception as e:
#         return jsonify({
#             "isSuccess": False,
#             "message": f"Embedding failed: {str(e)}"
#         }), 500

#     # Fetch embeddings from both tables
#     # users_normal = fetch_all_users_with_embeddings()  # From `users` table
#     # users_ref = fetch_all_ref_users_with_embeddings()
    
#     with cache_lock:
#         users_normal = users_normal_cache.copy()  # From `users` table
#         users_ref = users_ref_cache.copy()  # From `users_ref` table

#     best_user = None
#     best_distance = float("inf")
#     source = None  # 'users' or 'users_ref'

#     # Compare with users table
#     for user in users_normal:
#         distance = cosine(input_embedding, np.array(user['embedding']))
#         if distance < best_distance:
#             best_user = user
#             best_distance = distance
#             source = 'users'

#     # Compare with users_ref table
#     for user in users_ref:
#         distance = cosine(input_embedding, np.array(user['embedding']))
#         if distance < best_distance:
#             best_user = user
#             best_distance = distance
#             source = 'users_ref'

#     THRESHOLD = 0.4
#     if best_distance < THRESHOLD and best_user:
#         if source == 'users':
#             identity = {
#                 "username": best_user.get("username"),
#                 "email": best_user.get("email"),
#                 "mobile": best_user.get("mobile")
#             }
#         elif source == 'users_ref':
#             identity = {
#                 "ref_id": best_user.get("ref_id"),
#                 "ref_type": best_user.get("ref_type")
#             }
#         else:
#             identity = {"user_id": best_user.get("user_id")}

#         return jsonify({
#             "isSuccess": True,
#             "message": "User matched",
#             "identity": identity
#         })

#     else:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No matching user found"
#         }), 404







if __name__ == '__main__':
    load_embedding_caches()
    app.run(debug=True, host='0.0.0.0')
    

