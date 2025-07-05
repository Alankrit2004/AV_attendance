import pyodbc
from deepface import DeepFace
import json
import cv2

def connect_db():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=GlitchPC\\SQLEXPRESS;'
        'DATABASE=ATTENDACE;'
        'Trusted_Connection=yes;'
    )
    return conn

def get_next_label():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(label) FROM users")
    result = cursor.fetchone()
    conn.close()
    return (result[0] + 1) if result[0] is not None else 0

def fetch_all_users_with_embeddings():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, embedding, photo_path, email, mobile, ref_id, ref_type
        FROM users
    """)
    rows = cursor.fetchall()

    users = []
    for row in rows:
        user_id, username, embedding_json, photo_path, email, mobile, ref_id, ref_type = row
        try:
            embedding = json.loads(embedding_json)
        except Exception:
            embedding = None  # Skip invalid data

        if embedding:
            users.append({
                "user_id": user_id,
                "username": username,
                "embedding": embedding,
                "photo_path": photo_path,
                "email": email,
                "mobile": mobile,
                "ref_id": ref_id,
                "ref_type": ref_type
            })

    cursor.close()
    conn.close()
    return users






def get_username_by_label(label):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE label = ?", (label,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def register_user_to_db(username=None, email=None, mobile=None, image_path=None, ref_id=None, ref_type=None):
    conn = connect_db()
    cursor = conn.cursor()

    # Read image and extract embedding
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image at path: " + image_path)

    embedding_result = DeepFace.represent(img_path=image_path, model_name="SFace", enforce_detection=False)
    embedding = embedding_result[0]["embedding"]
    embedding_json = json.dumps(embedding)

    # --- Check if user exists ---
    if username:
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        exists = cursor.fetchone()[0] > 0
    elif ref_id and ref_type:
        cursor.execute("SELECT COUNT(*) FROM users WHERE ref_id = ? AND ref_type = ?", (ref_id, ref_type))
        exists = cursor.fetchone()[0] > 0
    else:
        raise Exception("Either username or (ref_id and ref_type) must be provided")

    # --- Update or Insert ---
    if exists:
        if username:
            cursor.execute("""
                UPDATE users SET email = ?, mobile = ?, photo_path = ?, embedding = ?, ref_id = ?, ref_type = ?
                WHERE username = ?
            """, (email, mobile, image_path, embedding_json, ref_id, ref_type, username))
        else:
            cursor.execute("""
                UPDATE users SET photo_path = ?, embedding = ?, username = ?, email = ?, mobile = ?
                WHERE ref_id = ? AND ref_type = ?
            """, (image_path, embedding_json, username, email, mobile, ref_id, ref_type))
    else:
        cursor.execute("""
            INSERT INTO users (username, email, mobile, photo_path, embedding, ref_id, ref_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, email, mobile, image_path, embedding_json, ref_id, ref_type))

    conn.commit()
    cursor.close()
    conn.close()




