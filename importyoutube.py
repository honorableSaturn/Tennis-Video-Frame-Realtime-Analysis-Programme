import yt_dlp, time, sys, os
# sys.path.append(r"R:\Coding\venv\")
url = input('')

counter = 1

while os.path.exists(f"downloaded_video{counter}.mp4"):
    counter += 1
path = f"downloaded_video{str(counter)}.mp4"

options = {
    'format': 'bestvideo[ext=mp4][height<=720][fps>=50]/bestvideo[ext=mp4][height<=720]/best[ext=mp4]',
    'outtmpl': path
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])
