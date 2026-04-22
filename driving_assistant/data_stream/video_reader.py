"""Video reader for extracting frames from video files."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Generator


class VideoReader:
    """Read frames from video files using OpenCV.

    Args:
        video_path: Path to the video file
    """

    def __init__(self, video_path):
        """Initialize the video reader.

        Args:
            video_path (str or Path): Path to the video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Opened video: {self.video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index.

        Args:
            frame_idx (int): Frame index (0-based)

        Returns:
            np.ndarray: Frame as BGR image, or None if failed to read
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.cap.read()
        return frame if success else None

    def get_frames_batch(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 1
    ) -> list:
        """Get a batch of frames.

        Args:
            start_frame (int): Starting frame index
            end_frame (int, optional): Ending frame index (inclusive). If None, read to end.
            skip_frames (int): Only return every nth frame

        Returns:
            list: List of frames as numpy arrays
        """
        if end_frame is None:
            end_frame = self.total_frames - 1

        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx <= end_frame:
            success, frame = self.cap.read()
            if not success:
                break

            frames.append(frame)
            frame_idx += skip_frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        return frames

    def iter_frames(self, skip_frames: int = 1) -> Generator:
        """Iterate through video frames.

        Args:
            skip_frames (int): Only yield every nth frame

        Yields:
            tuple: (frame_index, frame) where frame is BGR numpy array
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            if frame_idx % skip_frames == 0:
                yield frame_idx, frame

            frame_idx += 1

    def close(self):
        """Close the video file."""
        if self.cap:
            self.cap.release()
        print(f"Closed video: {self.video_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure video is closed."""
        if hasattr(self, 'cap'):
            self.close()
