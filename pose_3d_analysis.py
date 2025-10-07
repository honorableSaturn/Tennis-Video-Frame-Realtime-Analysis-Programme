import cv2, sys, time
import mediapipe as mp
import numpy as np
import pyautogui
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

screen_w, screen_h = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cv2.namedWindow("Pose 3D Analysis", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose 3D Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

def compute_3d_angles(coord1, coord2, coord3):
    ax, ay, az = coord1
    bx, by, bz = coord2
    cx, cy, cz = coord3
    ab = np.array([ax - bx, ay - by, az - bz])
    cb = np.array([cx - bx, cy - by, cz - bz])
    dot = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None
    cosine = dot / (norm_ab * norm_cb)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(np.arccos(cosine))

def compute_3d_coordinate_average(*args):
    if not args:
        return None, None, None
    arr = np.array(args)
    avg = np.mean(arr, axis=0)
    return tuple(avg)

def plot_pose_3d(landmarks):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*landmarks)
    ax.scatter(xs, ys, zs, c='r', marker='o')
    # Optionally, draw connections (bones)
    connections = mp_pose.POSE_CONNECTIONS
    for start, end in connections:
        x = [landmarks[start][0], landmarks[end][0]]
        y = [landmarks[start][1], landmarks[end][1]]
        z = [landmarks[start][2], landmarks[end][2]]
        ax.plot(x, y, z, c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Pose")
    plt.show(block=False)
    plt.pause(0.001)
    plt.close(fig)

def animate_3d_pose(landmarks_sequence):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        landmarks = landmarks_sequence[frame_idx]
        xs, ys, zs = zip(*landmarks)
        ax.scatter(xs, ys, zs, c='r', marker='o')
        connections = mp_pose.POSE_CONNECTIONS
        for start, end in connections:
            x = [landmarks[start][0], landmarks[end][0]]
            y = [landmarks[start][1], landmarks[end][1]]
            z = [landmarks[start][2], landmarks[end][2]]
            ax.plot(x, y, z, c='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Pose Frame {frame_idx+1}/{len(landmarks_sequence)}")
        ax.set_xlim(0, screen_w)
        ax.set_ylim(0, screen_h)
        ax.set_zlim(-screen_w//2, screen_w//2)

    ani = FuncAnimation(fig, update, frames=len(landmarks_sequence), interval=50, repeat=False)
    plt.show()

# Modify vfra_3d to collect landmarks and call animate_3d_pose at the end
def vfra_3d(path):
    visualize_angles = 1
    compute_frames = 1
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_delay = int(1000 / fps)
    counter = 0
    landmarks_sequence = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks and compute_frames == 1:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                      mp.solutions.drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=2))
        elif results.pose_landmarks and compute_frames == 0:
            print(f"landmarks skipped at {str(time.ctime())} [{counter}]")
        else:
            print(f"landmarks lost at {str(time.ctime())} [{counter}]")
            counter += 1

        if visualize_angles == 1 and results.pose_landmarks:
            h, w, _ = frame.shape
            landmark_3d_coords = [(lm.x * w, lm.y * h, lm.z * w) for lm in results.pose_landmarks.landmark]
            landmarks_sequence.append(landmark_3d_coords)

            right_arm_angle_3d = compute_3d_angles(landmark_3d_coords[12], landmark_3d_coords[14], landmark_3d_coords[16])
            left_arm_angle_3d = compute_3d_angles(landmark_3d_coords[11], landmark_3d_coords[13], landmark_3d_coords[15])

            annocolor = (0, 255, 0)
            if right_arm_angle_3d is not None:
                cv2.putText(frame,
                            f"Right Arm 3D: {round(right_arm_angle_3d, 2)}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            annocolor,
                            2)
            if left_arm_angle_3d is not None:
                cv2.putText(frame,
                            f"Left Arm 3D: {round(left_arm_angle_3d, 2)}",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            annocolor,
                            2)

        h, w = frame.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        top = (screen_h - new_h) // 2
        bottom = screen_h - new_h - top
        left = (screen_w - new_w) // 2
        right = screen_w - new_w - left

        bordered_frame = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        cv2.imshow("Pose 3D Analysis", bordered_frame)

        key = cv2.waitKey(frame_delay) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == 32:
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == 32:
                    break
                elif key2 == ord('q') or key2 == 27:
                    video.release()
                    cv2.destroyAllWindows()
                    sys.exit()
        elif key == ord('k'):
            compute_frames = (compute_frames + 1) % 2
        elif key == ord('o'):
            visualize_angles = (visualize_angles + 1) % 2

    video.release()
    cv2.destroyAllWindows()

    # Animate the collected 3D poses
    if landmarks_sequence:
        animate_3d_pose(landmarks_sequence)

if __name__ == "__main__":
    vfra_3d("downloaded_video2.mp4")