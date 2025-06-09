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
    cursor.execute("SELECT id AS user_id, username, email, mobilenumber, embedding, photo_path FROM users")
    rows = cursor.fetchall()

    users = []
    for row in rows:
        user_id, username, email, mobilenumber, embedding_json, photo_path = row
        try:
            embedding = json.loads(embedding_json)
        except Exception:
            embedding = None

        if embedding:
            users.append({
                "user_id": user_id,
                "username": username,
                "email": email,
                "mobilenumber": mobilenumber,
                "embedding": embedding,
                "photo_path": photo_path
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


def register_user_to_db(username, email, mobilenumber, image_path):
    conn = connect_db()
    cursor = conn.cursor()

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image at path: " + image_path)

    # Get embedding using DeepFace with SFace
    embedding_result = DeepFace.represent(img_path=image_path, model_name="SFace", enforce_detection=False)
    embedding = embedding_result[0]["embedding"]
    embedding_json = json.dumps(embedding)

    # Check if user already exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
    if cursor.fetchone()[0] > 0:
        # Update existing user's info
        cursor.execute(
            "UPDATE users SET email = ?, mobilenumber = ?, photo_path = ?, embedding = ? WHERE username = ?",
            (email, mobilenumber, image_path, embedding_json, username)
        )
    else:
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, email, mobilenumber, photo_path, embedding) VALUES (?, ?, ?, ?, ?)",
            (username, email, mobilenumber, image_path, embedding_json)
        )

    conn.commit()
    cursor.close()
    conn.close()



def log_attendance(username):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if row:
        user_id = row[0]
        cursor.execute("INSERT INTO attendance (user_id) VALUES (?)", (user_id,))
        conn.commit()
    cursor.close()
    conn.close()
