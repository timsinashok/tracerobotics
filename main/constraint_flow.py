import cv2
import numpy as np
import mediapipe as mp
import os
import glob

# --- CONFIGURATION ---
VIDEO_SOURCE = "videos/*.mp4"  # Path to video files
SEARCH_RADIUS = 150  # Only look for objects within 150px of the hand
CONSTRAINT_THRESHOLD = 0.9  # 90% motion correlation required
MIN_SPEED = 2.0  # If hand is not moving, we can't infer constraints

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# We use Sparse Optical Flow (Lucas-Kanade) for Phase 1 MVP
# It's faster than CoTracker for real-time testing
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

def get_hand_center(landmarks, w, h):
    """Returns average (x,y) of the wrist and middle finger base."""
    # Wrist is index 0, Middle Base is index 9
    x = int((landmarks.landmark[0].x + landmarks.landmark[9].x) / 2 * w)
    y = int((landmarks.landmark[0].y + landmarks.landmark[9].y) / 2 * h)
    return np.array([x, y], dtype=np.float32)

def process_video(video_path):
    """Process a single video file with constraint flow analysis."""
    print(f"\nProcessing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 'continue'
    
    # Get video properties
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {video_width}x{video_height}")
    
    # Create window with proper size (scale up small videos)
    window_name = 'Research Phase 1: Constraint Flow'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # If video is small, scale it up for better visibility
    if video_width < 640:
        scale_factor = 640 / video_width
        display_width = int(video_width * scale_factor)
        display_height = int(video_height * scale_factor)
    else:
        display_width = video_width
        display_height = video_height
    
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # State variables for Optical Flow
    old_gray = None
    p0 = None # Points we are tracking
    prev_hand_center = None
    frame_count = 0
    
    print(">>> Analyzing constraint field...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing {os.path.basename(video_path)} ({frame_count} frames)")
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy()
        h, w, _ = frame.shape

        # 1. DETECT HAND (The "Driver")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_velocity = np.array([0.0, 0.0])
        hand_present = False
        hand_center = None

        if results.multi_hand_landmarks:
            hand_present = True
            landmarks = results.multi_hand_landmarks[0]
            hand_center = get_hand_center(landmarks, w, h)
            
            # Draw skeleton for reference
            mp_draw.draw_landmarks(vis_frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # 2. OPTICAL FLOW (The "Physics")
        if old_gray is not None and hand_present:
            
            # A. If we lost points or just started, generate new points to track around the hand
            if p0 is None or len(p0) < 10:
                # Create a mask around the hand to find features to track
                roi_mask = np.zeros_like(old_gray)
                cv2.circle(roi_mask, (int(hand_center[0]), int(hand_center[1])), SEARCH_RADIUS, 255, -1)
                # Find good features to track (corners) within the radius
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

            if p0 is not None:
                # B. Calculate Optical Flow (Where did points move?)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # C. Calculate Hand Velocity (Approximate)
                    # We assume the hand center delta is the "Driver Velocity"
                    # (In Phase 2 we will use 3D MediaPipe velocity, this is 2D approx)
                    if prev_hand_center is not None:
                        hand_velocity = hand_center - prev_hand_center
                    
                    hand_speed = np.linalg.norm(hand_velocity)

                    # D. Constraint Classification
                    # Only analyze if hand is actually moving (otherwise everything is static)
                    if hand_speed > MIN_SPEED:
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            
                            # Point Velocity
                            point_vel = new - old
                            point_speed = np.linalg.norm(point_vel)
                            
                            # -- THE CORE LOGIC --
                            # Calculate Cosine Similarity
                            if point_speed > 0:
                                dot_product = np.dot(hand_velocity, point_vel)
                                similarity = dot_product / (hand_speed * point_speed)
                                
                                # Is the speed similar? (Rigid bodies move at same speed)
                                speed_ratio = min(hand_speed, point_speed) / max(hand_speed, point_speed)
                                
                                # If Direction is Same AND Speed is Same -> RIGID CONSTRAINT
                                if similarity > CONSTRAINT_THRESHOLD and speed_ratio > 0.7:
                                    # Draw Green Vector (Attached)
                                    cv2.arrowedLine(vis_frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
                                    cv2.circle(vis_frame, (int(a), int(b)), 3, (0, 255, 0), -1)
                                else:
                                    # Draw Red Dot (Independent / Background)
                                    cv2.circle(vis_frame, (int(a), int(b)), 2, (0, 0, 255), -1)

                    p0 = good_new.reshape(-1, 1, 2)

        # Update State
        old_gray = frame_gray.copy()
        if hand_present:
            prev_hand_center = hand_center

        # UI Overlay
        cv2.rectangle(vis_frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(vis_frame, "CONSTRAINT FLOW MVP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, os.path.basename(video_path), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow(window_name, vis_frame)

        # Use 30ms delay for video playback (33fps), or press 'q' to quit, 'n' for next video
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return 'quit'
        elif key == ord('n'):
            break

    cap.release()
    return 'continue'

# --- MAIN LOOP ---
# Get all video files
video_files = glob.glob(VIDEO_SOURCE)

if not video_files:
    print(f"No video files found matching pattern: {VIDEO_SOURCE}")
    print("Please check the path and try again.")
else:
    print(f"Found {len(video_files)} video file(s):")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    
    # Process each video
    for video_file in video_files:
        result = process_video(video_file)
        if result == 'quit':
            print("\nQuitting...")
            break
        # Small delay between videos
        cv2.waitKey(500)

cv2.destroyAllWindows()
print("\nAll videos processed!")