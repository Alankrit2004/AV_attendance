from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import os
from db import register_user_to_db
from deepface import DeepFace
from db import connect_db, fetch_all_users_with_embeddings
from scipy.spatial.distance import cosine
import datetime



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

@app.route('/register', methods=['POST'])
# @cross_origin()
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

        # Read image as color (DeepFace expects RGB or BGR)
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "isSuccess": False,
                "message": "Invalid image"
            }), 400

        # --- Preprocessing: denoise and blur ---
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # ---------------------------------------

        # Detect faces (using your existing face_cascade)
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({
                "isSuccess": False,
                "message": "No face detected"
            }), 400

        # Save full image or detected face region - here saving full image for simplicity
        user_folder = os.path.join(USERS_DIR, username)
        os.makedirs(user_folder, exist_ok=True)

        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)  # Save full color image

        # Store user info + image path in DB
        register_user_to_db(username, email, mobilenumber, image_path=save_path, ref_id=None, ref_type=None)  # Update your function to accept these params

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

@app.route('/register_v2', methods=['POST'])
def register_v2():
    print("Register V2 endpoint hit")
    try:
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

        # Read image as color
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "isSuccess": False,
                "message": "Invalid image"
            }), 400

        # Preprocessing: denoise and blur
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
        safe_name = f"{ref_id}"
        user_folder = os.path.join(USERS_DIR, safe_name)
        os.makedirs(user_folder, exist_ok=True)

        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)

        # Store user info in DB (reuse same function, update it if needed to handle ref_type logic)
        register_user_to_db(username=None, email=None, mobile=None, image_path=save_path, ref_id=ref_id, ref_type=ref_type)  # pass None for email and phone

        return jsonify({
            "isSuccess": True,
            "message": f"User '{ref_id}' registered successfully",
            "imagePath": save_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "isSuccess": False,
            "message": f"Server error: {str(e)}"
        }), 500

def log_attendance(user_id):
    try:
        conn = connect_db()
        cursor = conn.cursor()

        today = datetime.datetime.now().date()
        now = datetime.datetime.now()

        # Fetch user details
        cursor.execute("""
            SELECT username, email, mobile, ref_id, ref_type
            FROM users
            WHERE id = ?
        """, (user_id,))
        user_row = cursor.fetchone()

        if not user_row:
            return "user not found"

        username, email, mobile, ref_id, ref_type = user_row

        # Fetch latest attendance record for today
        cursor.execute("""
            SELECT id, check_in_time, check_out_time
            FROM attendance
            WHERE user_id = ? AND attendance_date = ?
            ORDER BY check_in_time DESC
        """, (user_id, today))
        row = cursor.fetchone()

        if row and row.check_out_time is None:
            # If previous record has no check-out → Update check_out_time
            cursor.execute("""
                UPDATE attendance
                SET check_out_time = ?
                WHERE id = ?
            """, (now, row.id))
            conn.commit()
            return "check-out"

        else:
            # Insert new check-in
            cursor.execute("""
                INSERT INTO attendance (
                    user_id, check_in_time, attendance_date,
                    username, email, mobile, ref_id, ref_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, now, today,
                username, email, mobile, ref_id, ref_type
            ))
            conn.commit()
            return "check-in"

    except Exception as e:
        return f"error: {str(e)}"
    finally:
        conn.close()






# @app.route('/recognize', methods=['POST'])
# # @cross_origin()
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

#     # --- Preprocessing: denoise and blur ---
#     input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
#     input_img = cv2.GaussianBlur(input_img, (3, 3), 0)
#     # ---------------------------------------

#     # Detect faces before embedding
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

#     users = fetch_all_users_with_embeddings()  # [{'user_id': ..., 'username': ..., 'embedding': [...], 'imagePath': ...}, ...]

#     try:
#         input_embedding = DeepFace.represent(img_path=input_img, model_name="SFace", enforce_detection=False)[0]['embedding']
#     except Exception as e:
#         return jsonify({
#             "isSuccess": False,
#             "message": f"Embedding failed: {str(e)}"
#         }), 500

#     best_user = None
#     best_distance = 9999

#     for user in users:
#         user_embedding = np.array(user['embedding'])
#         distance = cosine(input_embedding, user_embedding)
#         if distance < best_distance:
#             best_distance = distance
#             best_user = user

#     THRESHOLD = 0.4

#     if best_distance < THRESHOLD and best_user:
#         # Fetch email and mobile for the recognized user
#         email = best_user.get('email')
#         mobile = best_user.get('mobile')
#         status = log_attendance(best_user['user_id'], best_user['username'])
#         return jsonify({
#             "isSuccess": True,
#             "message": f"User '{best_user['username']}' recognized. Attendance: {status}",
#             "email": email,
#             "mobile": mobile
#         })
#     else:
#         return jsonify({
#             "isSuccess": False,
#             "message": "No matching user found"
#         }), 404


@app.route('/recognize', methods=['POST'])
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

    users = fetch_all_users_with_embeddings()  # [{'user_id', 'embedding', 'username', 'email', 'mobile', 'ref_id', 'ref_type'}]

    try:
        input_embedding = DeepFace.represent(img_path=input_img, model_name="SFace", enforce_detection=False)[0]['embedding']
    except Exception as e:
        return jsonify({
            "isSuccess": False,
            "message": f"Embedding failed: {str(e)}"
        }), 500

    best_user = None
    best_distance = float("inf")

    for user in users:
        user_embedding = np.array(user['embedding'])
        distance = cosine(input_embedding, user_embedding)
        if distance < best_distance:
            best_distance = distance
            best_user = user

    THRESHOLD = 0.4
    if best_distance < THRESHOLD and best_user:
        user_id = best_user.get("user_id")
        username = best_user.get("username")
        email = best_user.get("email")
        mobile = best_user.get("mobile")
        ref_id = best_user.get("ref_id")
        ref_type = best_user.get("ref_type")

        # Log attendance
        status = log_attendance(user_id)

        # Build identity object cleanly
        if username and (email or mobile):
            identity = {
                "username": username,
                "email": email,
                "mobile": mobile
            }
        elif ref_id and ref_type:
            identity = {
                "ref_id": ref_id,
                "ref_type": ref_type
            }
        else:
            identity = {
                "user_id": user_id
            }

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





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

