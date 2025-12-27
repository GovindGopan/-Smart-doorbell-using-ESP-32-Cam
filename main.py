from fastapi import FastAPI, Request, Body
from datetime import datetime
import os
import uvicorn
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import yagmail
from ultralytics import YOLO
import httpx
from pydantic import BaseModel

app = FastAPI()

IMAGES_FOLDER = "images"
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Face recognition setup
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load known faces embeddings
known_faces_path = "known_faces.pt"
if os.path.exists(known_faces_path):
    known_faces = torch.load(known_faces_path)
    print("Loaded known face embeddings")
else:
    known_faces = {}
    print("No known faces found. Add some!")

# Gmail setup
EMAIL_ID = os.getenv("EMAIL_ID")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # Use App Passwords for Gmail

if not EMAIL_ID or not EMAIL_PASSWORD:
    raise ValueError("EMAIL_ID and EMAIL_PASSWORD must be set as environment variables")

yag = yagmail.SMTP(EMAIL_ID, EMAIL_PASSWORD)

# YOLO model setup
yolo_model = YOLO('yolov8n.pt')
yolo_package_model= YOLO('best.pt')  # Ensure the model is loaded
# ESP32 setup
ESP32_IP = os.getenv("ESP32_IP", "http://192.168.1.6")  # Your ESP32 IP (replace if needed)
ESP32_BUZZER_ENDPOINT = f"{ESP32_IP}/buzzer"

class ESP32IPRequest(BaseModel):
    ip: str

class ESP32LogRequest(BaseModel):
    log: str

# Helper to send email
def send_email(img_np, alert_type="Unknown Person", detected_name=None):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if detected_name:
        subject = f"{detected_name} is at the door"
        contents = f"{detected_name} at {time_str}"
        filename = f"{detected_name}_{time_str}.jpg"
    else:
        subject = f"{alert_type} at the door"
        contents = f"{alert_type} at {time_str}"
        filename = f"{alert_type}_{time_str}.jpg"
    saved = cv2.imwrite(filename, img_np)
    if not saved:
        print("Failed to write image to disk:", filename)
        return
    try:
        yag.send(
             to=EMAIL_ID,
             subject=subject,
             contents=contents,
             attachments=[filename]
         )
        print(f"Email sent for {subject}")
    except Exception as e:
        print("Failed to send email:", e)
    finally:
        try:
            os.remove(filename)
        except Exception:
            pass

# Helper to trigger buzzer
async def trigger_buzzer(state="on"):
    try:
        async with httpx.AsyncClient() as client:
            params = {"state": state}
            response = await client.post(ESP32_BUZZER_ENDPOINT, params=params, timeout=5.0)
            print(f"Buzzer triggered: {response.text}")
    except Exception as e:
        import traceback
        print(f"Failed to trigger buzzer: {e}")
        traceback.print_exc()

@app.post("/setup-esp32-ip")
def setup_esp32_ip(data: ESP32IPRequest):
    global ESP32_IP, ESP32_BUZZER_ENDPOINT
    
    # FIXED: Ensure the IP has http:// protocol
    if not data.ip.startswith("http://") and not data.ip.startswith("https://"):
        ESP32_IP = f"http://{data.ip}"
    else:
        ESP32_IP = data.ip
    
    # Remove port :80 if present and rebuild endpoint
    ESP32_IP = ESP32_IP.replace(":80", "")
    ESP32_BUZZER_ENDPOINT = f"{ESP32_IP}/buzzer"
    
    print(f"ESP32 IP updated to: {ESP32_IP}")
    print(f"ESP32 Buzzer endpoint: {ESP32_BUZZER_ENDPOINT}")
    return {"message": f"ESP32 IP set to {ESP32_IP}"}

@app.post("/esp32-log")
def esp32_log(data: ESP32LogRequest):
    log_message = data.log
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ESP32 LOG {time_str}]: {log_message}")
    return {"message": "Log received", "log": log_message}

@app.post("/upload")
async def upload_image(request: Request):
    # Receive and decode the incoming image
    image_bytes = await request.body()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    detected_person_name = None
    save_path = None
    alert_type = None
    detected_object_classes = []
    is_object_detected = False

    # --- Object Detection ---
    print("Running object detection...")
    try:
        yolo_results = yolo_model(image, classes=[0])  # Detect only persons
        for result in yolo_results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                object_class = result.names[int(cls)]
                detected_object_classes.append(object_class)
                # Draw bounding box and label for each detected object
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, object_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                is_object_detected = True
        print(f"Detected objects: {detected_object_classes}")

        package_model_results = yolo_package_model(image)  # For future use if needed
        package_detected = False
        for result in package_model_results:
            for cls in result.boxes.cls:
                if yolo_package_model.names[int(cls)] == 'package':
                    package_detected = True
                    detected_object_classes.append('package')
                    is_object_detected = True
                    break
            if package_detected:
                print("Package detected by package model")
                break

    except Exception as e:
        print("Object detection failed:", e)
        is_object_detected = False

    # --- Save if no object detected ---
    if not is_object_detected:
        nothing_folder = "nothingdetected"
        os.makedirs(nothing_folder, exist_ok=True)
        save_path = os.path.join(nothing_folder, f"nothing_{timestamp}.jpg")
        cv2.imwrite(save_path, image)
        print(f"No object detected. Saved image: {save_path}")
        return {"message": "No object detected", "filename": save_path, "alert": "None", "person": None}

    # --- If person detected, run face detection ---
    if "person" in detected_object_classes:
        print("Person detected. Running face detection...")
        face_boxes, face_probs = mtcnn.detect(image)
        is_known_person = False
        if face_boxes is not None:
            print(f"Faces detected: {len(face_boxes)}")
            for box in face_boxes:
                box = [int(b) for b in box]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(image.shape[1], box[2]), min(image.shape[0], box[3])
                if x2 > x1 and y2 > y1:
                    face_crop = image[y1:y2, x1:x2]
                    print("Face crop shape:", face_crop.shape)
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    face_tensor = mtcnn(face_pil)
                    if face_tensor is not None:
                        face_embedding = resnet(face_tensor.unsqueeze(0)).detach()
                        for name, known_embedding in known_faces.items():
                            if (face_embedding - known_embedding).norm().item() < 0.8:
                                is_known_person = True
                                detected_person_name = name
                                print(f"Recognized known person: {name}")
                                break
                    else:
                        print(f"Face tensor is None for crop: ({x1}, {y1}, {x2}, {y2})")
            if is_known_person:
                person_folder = "persondetected"
                os.makedirs(person_folder, exist_ok=True)
                save_path = os.path.join(person_folder, f"person_detected_{timestamp}_with_face.jpg")
                cv2.imwrite(save_path, image)
                print(f"Known person detected. Saved image: {save_path}")
                # Run buzzer and email in background
                import asyncio
                asyncio.create_task(trigger_buzzer("on"))
                asyncio.create_task(asyncio.to_thread(send_email, image, alert_type="Known Person", detected_name=detected_person_name))
                return {"message": "Known person detected", "filename": save_path, "alert": "Known Person", "person": detected_person_name}
            else:
                person_folder = "persondetected"
                os.makedirs(person_folder, exist_ok=True)
                save_path = os.path.join(person_folder, f"person_detected_{timestamp}_no_face.jpg")
                cv2.imwrite(save_path, image)
                print(f"Person detected but no recognized face. Saved image: {save_path}")
                import asyncio
                asyncio.create_task(trigger_buzzer("on"))
                asyncio.create_task(asyncio.to_thread(send_email, image, alert_type="Unknown Person"))
                return {"message": "Unknown person detected", "filename": save_path, "alert": "Unknown Person", "person": None}
        else:
            print("Person detected but no face found.")
            person_folder = "persondetected"
            os.makedirs(person_folder, exist_ok=True)
            save_path = os.path.join(person_folder, f"person_detected_{timestamp}_no_face.jpg")
            cv2.imwrite(save_path, image)
            import asyncio
            asyncio.create_task(trigger_buzzer("on"))
            asyncio.create_task(asyncio.to_thread(send_email, image, alert_type="Person Detected But No Face"))
            return {"message": "Person detected but no face found", "filename": save_path, "alert": "Person Detected But No Face", "person": None}
    
    elif "package" in detected_object_classes:
                package_folder = "packagedetected"
                os.makedirs(package_folder, exist_ok=True)
                save_path = os.path.join(package_folder, f"package_detected_{timestamp}.jpg")
                cv2.imwrite(save_path, image)
                print(f"Package detected . Saved image: {save_path}")
                import asyncio
                asyncio.create_task(trigger_buzzer("on"))
                asyncio.create_task(asyncio.to_thread(send_email, image, alert_type="Package Detected"))
        return {"message": "Package detected", "filename": save_path, "alert": "Package Detected", "person": None}
    
    
    else:
        # --- Other object detected ---
        object_folder = "objectdetected"
        os.makedirs(object_folder, exist_ok=True)
        save_path = os.path.join(object_folder, f"object_{'_'.join(detected_object_classes)}_{timestamp}.jpg")
        cv2.imwrite(save_path, image)
        print(f"Object(s) detected: {', '.join(detected_object_classes)}. Saved image: {save_path}")
        import asyncio
        asyncio.create_task(asyncio.to_thread(send_email, image, alert_type=f"Object Detected: {', '.join(detected_object_classes)}"))
        return {"message": f"Object(s) detected: {', '.join(detected_object_classes)}", "filename": save_path, "alert": "Object Detected", "person": None}    

@app.post("/buzzer-log")
async def log_buzzer_log():
    print("IR buzzer activated")
    return {"message": "IR buzzer activated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)