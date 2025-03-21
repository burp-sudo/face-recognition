from flask import Flask, render_template, request, redirect, Response, url_for
import sqlite3
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import base64
from flask import send_from_directory


app = Flask(__name__)

@app.route('/dataset/<path:filename>')
def serve_image(filename):
    return send_from_directory('dataset', filename)

# Ensure dataset folder exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            stream TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()

# Load student faces from DB
def load_known_faces():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name, image_path FROM students")
    rows = c.fetchall()
    conn.close()

    encodings = []
    names = []
    ids = []

    for student_id, name, img_path in rows:
        if os.path.exists(img_path):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(name)
                ids.append(student_id)

    return ids, names, encodings

# Mark attendance if not already present today
def mark_attendance(student_id):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE student_id = ? AND date = ?", (student_id, today))
    already_marked = c.fetchone()
    if not already_marked:
        c.execute("INSERT INTO attendance (student_id, date) VALUES (?, ?)", (student_id, today))
        conn.commit()
    conn.close()

# Stream webcam with face recognition
def gen_frames():
    known_ids, known_names, known_encodings = load_known_faces()
    cap = cv2.VideoCapture(0)

    process_this_frame = True

    while True:
        success, frame = cap.read()
        if not success:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match = np.argmin(face_distances)

                name = "Unknown"
                if matches[best_match]:
                    student_id = known_ids[best_match]
                    name = known_names[best_match]
                    mark_attendance(student_id)

                face_names.append(name)

        process_this_frame = not process_this_frame  # Toggle frame processing

        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Webcam feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Register new student
import base64

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        stream = request.form['stream']
        image_data = request.form['image_data']

        if image_data:
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            filename = f"{name.replace(' ', '_')}.jpg"
            image_path = f"dataset/{filename}"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO students (name, stream, image_path) VALUES (?, ?, ?)",
                      (name, stream, image_path))
            conn.commit()
            conn.close()

        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/students')
def students():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, name, stream, image_path FROM students")
    students = c.fetchall()
    conn.close()
    return render_template('students.html', students=students)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Get the image path first
    c.execute("SELECT image_path FROM students WHERE id = ?", (student_id,))
    row = c.fetchone()
    if row:
        image_path = row[0]
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete from students
    c.execute("DELETE FROM students WHERE id = ?", (student_id,))

    # Optionally: clean attendance too
    c.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))

    conn.commit()
    conn.close()

    return redirect(url_for('students'))


# Run the app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
