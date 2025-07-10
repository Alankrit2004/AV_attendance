from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from deepface import DeepFace
from db import register_user_to_db, register_user_ref_to_db
from db import fetch_all_ref_users_with_embeddings, fetch_all_users_with_embeddings
from db import log_attendance_ref, log_attendance_user
from scipy.spatial.distance import cosine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS_DIR = "users"
MODEL_PATH = "trained_model/model.yml"
LABEL_MAP_PATH = "trained_model/label_map.txt"
CONFIDENCE_THRESHOLD = 80

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs("trained_model", exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.post("/register_face")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    mobilenumber: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        if not username:
            raise HTTPException(status_code=400, detail="Username is required")
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        if not mobilenumber:
            raise HTTPException(status_code=400, detail="Mobile number is required")
        if not image:
            raise HTTPException(status_code=400, detail="No image file uploaded")

        img_array = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        user_folder = os.path.join(USERS_DIR, username)
        os.makedirs(user_folder, exist_ok=True)
        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)

        try:
            register_user_to_db(
                username=username,
                email=email,
                mobile=mobilenumber,
                image_path=save_path
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"isSuccess": True, "message": f"User '{username}' registered successfully", "imagePath": save_path}
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Server error: {str(e)}"})

@app.post("/register_face_v2")
async def register_v2(
    ref_id: str = Form(...),
    ref_type: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        if not ref_id:
            raise HTTPException(status_code=400, detail="ref_id is required")
        if not ref_type:
            raise HTTPException(status_code=400, detail="ref_type is required")
        if not image:
            raise HTTPException(status_code=400, detail="No image file uploaded")

        img_array = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        safe_folder = f"{ref_type}_{ref_id}".replace(" ", "_")
        user_folder = os.path.join(USERS_DIR, safe_folder)
        os.makedirs(user_folder, exist_ok=True)
        save_path = os.path.join(user_folder, "1.jpg")
        cv2.imwrite(save_path, img)

        try:
            register_user_ref_to_db(ref_id, ref_type, save_path)
        except Exception as reg_err:
            raise HTTPException(status_code=400, detail=str(reg_err))

        return {"isSuccess": True, "message": f"User '{ref_id}' registered successfully", "imagePath": save_path}
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Server error: {str(e)}"})

@app.post("/recognize_face")
async def recognize(image: UploadFile = File(...)):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image uploaded")
        img_array = np.frombuffer(await image.read(), np.uint8)
        input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if input_img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
        input_img = cv2.GaussianBlur(input_img, (3, 3), 0)
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")
        try:
            input_embedding = DeepFace.represent(
                img_path=input_img, model_name="SFace", enforce_detection=False
            )[0]['embedding']
        except Exception as e:
            return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Embedding failed: {str(e)}"})
        users_normal = fetch_all_users_with_embeddings()
        users_ref = fetch_all_ref_users_with_embeddings()
        best_user = None
        best_distance = float("inf")
        source = None
        for user in users_normal:
            distance = cosine(input_embedding, np.array(user['embedding']))
            if distance < best_distance:
                best_user = user
                best_distance = distance
                source = 'users'
        for user in users_ref:
            distance = cosine(input_embedding, np.array(user['embedding']))
            if distance < best_distance:
                best_user = user
                best_distance = distance
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
            return {
                "isSuccess": True,
                "message": f"User recognized. Attendance: {status}",
                "identity": identity,
                "status": status
            }
        else:
            return JSONResponse(status_code=404, content={"isSuccess": False, "message": "No matching user found"})
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Server error: {str(e)}"})

@app.post("/recognize_face_v2")
async def recognize_v2(image: UploadFile = File(...)):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image uploaded")
        img_array = np.frombuffer(await image.read(), np.uint8)
        input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if input_img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        input_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)
        input_img = cv2.GaussianBlur(input_img, (3, 3), 0)
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")
        try:
            input_embedding = DeepFace.represent(
                img_path=input_img, model_name="SFace", enforce_detection=False
            )[0]['embedding']
        except Exception as e:
            return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Embedding failed: {str(e)}"})
        users_normal = fetch_all_users_with_embeddings()
        users_ref = fetch_all_ref_users_with_embeddings()
        best_user = None
        best_distance = float("inf")
        source = None
        for user in users_normal:
            distance = cosine(input_embedding, np.array(user['embedding']))
            if distance < best_distance:
                best_user = user
                best_distance = distance
                source = 'users'
        for user in users_ref:
            distance = cosine(input_embedding, np.array(user['embedding']))
            if distance < best_distance:
                best_user = user
                best_distance = distance
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
            return {
                "isSuccess": True,
                "message": "User matched",
                "identity": identity
            }
        else:
            return JSONResponse(status_code=404, content={"isSuccess": False, "message": "No matching user found"})
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"isSuccess": False, "message": f"Server error: {str(e)}"})

# To run: uvicorn app_new:app --reload --host 0.0.0.0 --port 8000
