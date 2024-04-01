import cv2
import numpy as np
from keras.models import model_from_json
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
from fastapi import FastAPI, HTTPException

app = FastAPI()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize dictionary
emotion_durations = defaultdict(float)

start_time = None
current_emotion = None

# Create directory
recorded_video_dir = "recorded_videos"
os.makedirs(recorded_video_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Main route
@app.get("/emotion-detection")
async def capture_emotions():
    global start_time, current_emotion, out

    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        raise HTTPException(status_code=500, detail="Unable to open webcam")

    try:
        video_file = os.path.join(recorded_video_dir, f"recorded_video_{time.strftime('%Y%m%d%H%M%S')}.avi")
        out = cv2.VideoWriter(video_file, fourcc, 20.0, (640, 480))

        while True:
            ret, frame = webcam.read()
            
            if ret:
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    
                    face_roi = gray[y:y+h, x:x+w]

                    face_roi_resized = cv2.resize(face_roi, (48, 48))

                    img = extract_features(face_roi_resized)
                    pred = model.predict(img)
                    prediction_label = np.argmax(pred)

                    if current_emotion is not None and current_emotion != labels[prediction_label]:
                        emotion_durations[current_emotion] += time.time() - start_time
                        print(f"Emotion: {current_emotion}, Duration: {emotion_durations[current_emotion]:.2f} seconds")

                    if current_emotion != labels[prediction_label]:
                        start_time = time.time()
                        current_emotion = labels[prediction_label]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    cv2.putText(frame, labels[prediction_label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                out.write(frame)

                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()

        if out is not None:
            out.release()

        # Log the last detected emotion duration
        if current_emotion is not None:
            emotion_durations[current_emotion] += time.time() - start_time
            print(f"Emotion: {current_emotion}, Duration: {emotion_durations[current_emotion]:.2f} seconds")

        return {"emotion_durations": dict(emotion_durations)}
@app.get("/radar-graph")
async def generate_radar_graph():
    emotions = list(emotion_durations.keys())
    durations = list(emotion_durations.values())

    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, durations, color='blue', alpha=0.25)
    ax.plot(angles, durations, color='blue', linewidth=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(emotions)

    plt.show()

    return {"message": "Radar graph generated successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)

