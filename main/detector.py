import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # Use 0 for Webcam, or put a path like "video.mp4"
CONFIDENCE_THRESHOLD = 0.5 # YOLO confidence
IOU_THRESHOLD = 0.1 # How much overlap counts as "contact"? (10%)

# --- LOAD MODELS ---
print("Loading Models...")
# 1. Load YOLOv8 (Small version for speed)
# It will download 'yolov8n.pt' automatically on first run
object_model = YOLO("yolov8n.pt") 

# 2. Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def calculate_iou(boxA, boxB):
    """
    Calculate intersection area between hand and object.
    box = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # For Phase 1, we just care if there is significant Intersection
    # We compare intersection area to the Hand's area
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    
    if boxAArea == 0: return 0
    return interArea / boxAArea

# --- MAIN LOOP ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape

    # 1. OBJECT DETECTION (YOLO)
    # We only care about specific classes? For now, let's detect everything.
    # classes=[39, 41, ...]  # You can filter for specific objects like bottles/cups later
    yolo_results = object_model(frame, stream=True, verbose=False)

    object_boxes = []
    
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf)
            cls = int(box.cls)
            label = object_model.names[cls]

            # Filter out people (class 0) so we don't detect "contact" with ourselves
            if cls != 0 and conf > CONFIDENCE_THRESHOLD:
                object_boxes.append({'box': [x1, y1, x2, y2], 'label': label})
                # Draw Blue Box for Objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 2. HAND DETECTION (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    contact_detected = False
    status_text = "STATE: FREE MOTION"
    status_color = (0, 255, 0) # Green

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Calculate Hand Bounding Box
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            # Add some padding to hand box
            padding = 20
            hand_box = [x_min-padding, y_min-padding, x_max+padding, y_max+padding]

            # Check collision with ALL objects
            for obj in object_boxes:
                overlap = calculate_iou(hand_box, obj['box'])
                
                if overlap > IOU_THRESHOLD:
                    contact_detected = True
                    status_text = f"STATE: CONTACT -> {obj['label']}"
                    status_color = (0, 0, 255) # Red
                    
                    # Highlight the specific object
                    ox1, oy1, ox2, oy2 = obj['box']
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 4)

            # Draw Hand Box
            cv2.rectangle(frame, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), status_color, 2)
            
            # Optional: Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 3. DISPLAY UI
    # Create a dashboard overlay
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow('Phase 1: Contact Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()