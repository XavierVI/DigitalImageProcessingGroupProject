import os
import json
import shutil
import subprocess

video_dir = './data/reddit_dashcam_videos/'  # Or your symlinked drive path
annotation_file = 'labels.json'
annotations = {}

# Load existing progress if you have to stop halfway
if os.path.exists(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mkv'))]

def _play_with_external_player(path: str, window_name: str) -> bool:
    players = [
        (
            'ffplay',
            [
                'ffplay',
                '-autoexit',
                '-hide_banner',
                '-loglevel',
                'error',
                '-window_title',
                window_name,
            ],
        ),
        ('mpv', ['mpv', '--force-window=yes', '--title', window_name]),
        ('vlc', ['vlc', '--play-and-exit', '--quiet']),
    ]

    for binary, base_cmd in players:
        if shutil.which(binary):
            print(f"Using {binary} for playback. Close the player (or press 'q') to continue.")
            try:
                subprocess.run(base_cmd + [path], check=False)
                return True
            except OSError as exc:
                print(f"Failed to start {binary}: {exc}")

    return False


def play_video(path: str, window_name: str = 'Annotator') -> None:
    if _play_with_external_player(path, window_name):
        return

    # Fall back to OpenCV if no external player is available.
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print('No video player found. Install ffmpeg (ffplay), mpv, vlc, or opencv-python.')
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open video: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 33
    paused = False

    print("OpenCV controls: space=pause/resume, n=next video, q=quit playback")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


for vid in videos:
    if vid in annotations:
        continue

    print(f"Annotating: {vid}")
    play_video(os.path.join(video_dir, vid), window_name=vid)

    # Just type keywords like: red_light, night, truck
    tags = input("Enter keywords (comma-separated) or 's' to skip: ")

    if tags.lower() != 's':
        annotations[vid] = [t.strip().lower() for t in tags.split(',')]

        # Save after every video so you don't lose work
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)

print("All done! Labels saved to:", annotation_file)
