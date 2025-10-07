from ffpyplayer.player import MediaPlayer
import cv2

# Example usage: play video and audio together

def play_video_with_audio(video_path):
    cap = cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)

    while True:
        ret, frame = cap.read()
        audio_frame, val = player.get_frame()
        if not ret:
            break
        cv2.imshow('Video with Audio', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    player.close()

if __name__ == "__main__":
    play_video_with_audio("hittingwliamC.mp4")
