import cv2, csv, sys
import mediapipe as mp
import numpy as np
import math
import pyautogui


def writetofile(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        with open(f"{filename}_pose_landmarks.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "landmark_id", "x", "y", "z", "visibility"])
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        writer.writerow([frame_idx, i, lm.x, lm.y, lm.z, lm.visibility])
                frame_idx += 1
    cap.release()

#Video Frame Realtime Annotation

def vfra(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    canvas_width, canvas_height = 1920, 1080  # fixed 16:9 landscape
    cv2.namedWindow("VFRA Pose", cv2.WINDOW_AUTOSIZE)

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Create black canvas
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Scale original frame to fit canvas while preserving aspect ratio
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Center frame on canvas
                x_offset = (canvas_width - new_w) // 2
                y_offset = (canvas_height - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized

                # Map landmarks to scaled & padded coordinates
                scaled_landmarks = []
                for lm in landmarks:
                    x = int(lm.x * new_w) + x_offset
                    y = int(lm.y * new_h) + y_offset
                    scaled_landmarks.append((x, y))

                # Overlay for semi-transparent drawing
                overlay = canvas.copy()

                # Draw connections
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    cv2.line(overlay, scaled_landmarks[start_idx], scaled_landmarks[end_idx],
                             (0, 0, 255), 2)

                # Draw landmarks
                for x, y in scaled_landmarks:
                    cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

                # Annotate joint angles (elbows and knees example)
                joints = [(12,14,16), (11,13,15), (24,26,28), (23,25,27)]
                for a_idx, b_idx, c_idx in joints:
                    a = scaled_landmarks[a_idx]
                    b = scaled_landmarks[b_idx]
                    c = scaled_landmarks[c_idx]
                    angle = calculate_angle(a, b, c)
                    cv2.line(overlay, b, a, (255,255,0), 2)
                    cv2.line(overlay, b, c, (255,255,0), 2)
                    cv2.putText(overlay, f"{int(angle)}°", (b[0]+5, b[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                # Blend overlay with canvas for opacity effect
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            cv2.imshow("VFRA Pose", canvas)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # pause
                while True:
                    key2 = cv2.waitKey(30) & 0xFF
                    if key2 == 32:  # resume
                        break
                    elif key2 == ord('q') or key2 == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

    cap.release()
    cv2.destroyAllWindows()

def vfra_landscape(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose() as pose:
        # Fixed landscape resolution
        width, height = 1920, 1080  # 16:9
        cv2.namedWindow("Pose", cv2.WINDOW_AUTOSIZE)  # fixed size window

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw pose on original frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(181, 0, 15), thickness=2, circle_radius=1)
                )

            # Create black canvas
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Compute scaling to fit landscape while preserving aspect ratio
            h, w = frame.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize frame
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Center frame on canvas
            x_offset = (width - new_w) // 2
            y_offset = (height - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized

            cv2.imshow("Pose", canvas)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # pause
                while True:
                    key2 = cv2.waitKey(30) & 0xFF
                    if key2 == 32:
                        break
                    elif key2 == ord('q') or key2 == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

    cap.release()
    cv2.destroyAllWindows()

def calculate_angle(a, b, c):
    # a, b, c are (x, y) tuples; angle at point b
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def calculate_angle(a, b, c):
    # a, b, c are (x, y) tuples; angle at point b
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def vfra_wireframe_angles_padded(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose() as pose:
        # Fixed landscape resolution
        canvas_width, canvas_height = 1920, 1080  # 16:9
        cv2.namedWindow("Wireframe Angles", cv2.WINDOW_AUTOSIZE)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Black canvas
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Compute scaling to fit canvas while preserving aspect ratio
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w*scale), int(h*scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Compute offsets for centering
                x_offset = (canvas_width - new_w) // 2
                y_offset = (canvas_height - new_h) // 2

                # Map landmarks to scaled & padded coordinates
                scaled_landmarks = []
                for lm in landmarks:
                    x = int(lm.x * new_w) + x_offset
                    y = int(lm.y * new_h) + y_offset
                    scaled_landmarks.append((x,y))

                # Draw connections
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    cv2.line(canvas, scaled_landmarks[start_idx], scaled_landmarks[end_idx],
                             (0,0,255), 2)

                # Draw landmarks
                for x,y in scaled_landmarks:
                    cv2.circle(canvas, (x,y), 3, (0,255,0), -1)

                # Example: annotate elbow and knee angles
                joints = [(12,14,16), (11,13,15), (24,26,28), (23,25,27)]
                for a_idx, b_idx, c_idx in joints:
                    a = scaled_landmarks[a_idx]
                    b = scaled_landmarks[b_idx]
                    c = scaled_landmarks[c_idx]
                    angle = calculate_angle(a,b,c)
                    # Draw simple arc as two lines
                    cv2.line(canvas, b, a, (255,255,0), 2)
                    cv2.line(canvas, b, c, (255,255,0), 2)
                    # Annotate angle
                    cv2.putText(canvas, f"{int(angle)}°", (b[0]+5,b[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,90,0), 2)

            cv2.imshow("Wireframe Angles", canvas)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # pause
                while True:
                    key2 = cv2.waitKey(30) & 0xFF
                    if key2 == 32:
                        break
                    elif key2 == ord('q') or key2 == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

    cap.release()
    cv2.destroyAllWindows()




vfra("output.mp4")