from deepface import DeepFace
from scipy.spatial.distance import cosine
import json
import cv2
import datetime
from database import connect_db



def get_next_label():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(label) FROM users")
    result = cursor.fetchone()
    conn.close()
    return (result[0] + 1) if result[0] is not None else 0


def get_username_by_label(label):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE label = ?", (label,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def register_user_to_db(username, email, mobile, image_path):
    conn = connect_db()
    cursor = conn.cursor()

    # Read image and extract embedding
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image at path: " + image_path)

    embedding_result = DeepFace.represent(img_path=image_path, model_name="SFace", enforce_detection=False)
    embedding = embedding_result[0]["embedding"]
    embedding_json = json.dumps(embedding)

    # --- Check if username already exists ---
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
    if cursor.fetchone()[0] > 0:
        raise Exception(f"User with username '{username}' already exists.")

    # --- Check if face already exists ---
    face_exists, reason = is_face_already_registered(embedding)
    if face_exists:
        raise Exception("User already registered with this face")

    # --- Insert new user ---
    cursor.execute("""
        INSERT INTO users (username, email, mobile, photo_path, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (username, email, mobile, image_path, embedding_json))

    conn.commit()
    cursor.close()
    conn.close()




def register_user_ref_to_db(ref_id, ref_type, image_path):
    conn = connect_db()
    cursor = conn.cursor()

    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image at: " + image_path)

    embedding = DeepFace.represent(img_path=image_path, model_name="SFace", enforce_detection=False)[0]["embedding"]
    embedding_json = json.dumps(embedding)

    # Check for duplicate face
# ...existing code...
    face_exists, reason = is_face_already_registered(embedding)
    if face_exists:
        raise Exception("User already registered with this face")
# ...existing code...

    cursor.execute("SELECT COUNT(*) FROM users_ref WHERE ref_id = ?", (ref_id,))
    exists = cursor.fetchone()[0] > 0

    if exists:
        raise Exception(f"User with ref_id '{ref_id}' already exists")

    cursor.execute(
        "INSERT INTO users_ref (ref_id, ref_type, photo_path, embedding) VALUES (?, ?, ?, ?)",
        (ref_id, ref_type, image_path, embedding_json)
    )

    conn.commit()
    cursor.close()
    conn.close()




def is_face_already_registered(new_embedding, threshold=0.4):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT username, embedding FROM users")
    for username, emb_json in cursor.fetchall():
        try:
            existing_emb = json.loads(emb_json)
            if cosine(new_embedding, existing_emb) < threshold:
                return True, f"username: {username}"
        except:
            continue

    cursor.execute("SELECT ref_id, ref_type, embedding FROM users_ref")
    for ref_id, ref_type, emb_json in cursor.fetchall():
        try:
            existing_emb = json.loads(emb_json)
            if cosine(new_embedding, existing_emb) < threshold:
                return True, f"ref_id: {ref_id}, ref_type: {ref_type}"
        except:
            continue

    cursor.close()
    conn.close()
    return False, None

def fetch_all_ref_users_with_embeddings():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT id, ref_id, ref_type, embedding, photo_path FROM users_ref")
    rows = cursor.fetchall()

    users = []
    for row in rows:
        user_id, ref_id, ref_type, embedding_json, photo_path = row
        try:
            embedding = json.loads(embedding_json)
        except Exception:
            embedding = None

        if embedding:
            users.append({
                "user_id": user_id,
                "ref_id": ref_id,
                "ref_type": ref_type,
                "embedding": embedding,
                "photo_path": photo_path
            })

    cursor.close()
    conn.close()
    return users


def fetch_all_users_with_embeddings():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, embedding, photo_path, email, mobile
        FROM users
    """)
    rows = cursor.fetchall()

    users = []
    for row in rows:
        user_id, username, embedding_json, photo_path, email, mobile = row
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
                "mobile": mobile
            })

    cursor.close()
    conn.close()
    return users

def log_attendance_user(user_id):
    try:
        conn = connect_db()
        cursor = conn.cursor()

        now = datetime.datetime.now()
        today = now.date()

        # Fetch user details
        cursor.execute("""
            SELECT username, email, mobile
            FROM users
            WHERE id = ?
        """, (user_id,))
        user_row = cursor.fetchone()

        if not user_row:
            return "user not found"

        username, email, mobile = user_row

        # Insert new attendance entry (no duplicate check — allow multiple check-ins)
        cursor.execute("""
            INSERT INTO attendance (
                user_id, check_in_time, attendance_date,
                username, email, mobile
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, now, today,
            username, email, mobile
        ))

        conn.commit()
        return "check-in"

    except Exception as e:
        return f"error: {str(e)}"
    finally:
        conn.close()


def log_attendance_ref(ref_id):
    try:
        conn = connect_db()
        cursor = conn.cursor()

        now = datetime.datetime.now()
        today = now.date()

        # Fetch user details
        cursor.execute("""
            SELECT ref_type
            FROM users_ref
            WHERE ref_id = ?
        """, (ref_id,))
        user_row = cursor.fetchone()

        if not user_row:
            return "ref user not found"

        ref_type = user_row[0]

        # Insert new attendance entry (no duplicate check — allow multiple check-ins)
        cursor.execute("""
            INSERT INTO attendance (
                ref_id, ref_type, check_in_time, attendance_date
            )
            VALUES (?, ?, ?, ?)
        """, (
            ref_id, ref_type, now, today
        ))

        conn.commit()
        return "check-in"

    except Exception as e:
        return f"error: {str(e)}"
    finally:
        conn.close()

def update_user_ref_in_db(ref_id, ref_type, image_path):
    conn = connect_db()
    cursor = conn.cursor()

    # Optionally, update the embedding as well
    import cv2
    import json
    from deepface import DeepFace

    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read image at: " + image_path)

    embedding = DeepFace.represent(img_path=image_path, model_name="SFace", enforce_detection=False)[0]["embedding"]
    embedding_json = json.dumps(embedding)

    cursor.execute("""
        UPDATE users_ref
        SET ref_type = ?, photo_path = ?, embedding = ?
        WHERE ref_id = ?
    """, (ref_type, image_path, embedding_json, ref_id))

    if cursor.rowcount == 0:
        raise Exception(f"No user found with ref_id '{ref_id}'")

    conn.commit()
    cursor.close()
    conn.close()