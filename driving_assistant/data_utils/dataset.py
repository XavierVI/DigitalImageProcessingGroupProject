"""Traffic dataset loader for frames and videos."""

import os
from pathlib import Path
from PIL import Image
import torch
import cv2

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        This dataset is designed to return video frames from
        a video file. Each index is interpreted as a different
        video.

        @param root_dir: directory containing the videos.
        @param is_frames: if root_dir contains images of frames.
        """
        self.root_dir = root_dir
        self.transform = transform

        frame_types = ('.mp4', '.avi', '.mov')

        self.files = [
            f for f in os.listdir(root_dir)
            if f.endswith(frame_types)
        ]
        # current video which is open
        self.curr_video = None

    def step(self):
        """
        Fetch the next frame
        """
        if self.curr_video is None:
            return False, None

        success, frame = self.curr_video.read()

        if not success:
            self.curr_video.release()
            self.curr_video = None
            return success, None

        if self.transform:
            frame = self.transform(frame)

        return success, frame

    def get_video_name(self, idx):
        """
        Get the name of the video at index idx.
        """
        return self.files[idx]

    def get_current_time(self) -> float:
        """Returns current position in seconds."""
        if self.curr_video is None:
            return 0.0
        return self.curr_video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def __len__(self):
        """
        Can we get the number of frames in the
        video?
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        We could either return the video stream object
        or we could just use this to open a video and
        fetch inividual frames.

        Perhaps we can return the first frame?
        """
        # close the video stream
        if self.curr_video is not None:
            self.curr_video.release()

        video_path = os.path.join(self.root_dir, self.files[idx])
        # use opencv2 to open the video
        self.curr_video = cv2.VideoCapture(video_path)

        # check if video opened successfully
        is_opened = True

        if not self.curr_video.isOpened():
            is_opened = False

        # read the first frame
        _, frame = self.step()

        return is_opened, frame



class TrafficDataset(torch.utils.data.Dataset):
    """Dataset for loading traffic-related frames and videos.

    This dataset supports loading individual frames from a directory.
    This was purely used for the old Kaggle dataset we likely
    won't be using anymore.

    Args:
        root_dir (str or Path): Root directory containing frames or videos
        transform (callable, optional): Optional transforms to apply to images
        is_frames (bool): If True, loads frames; if False, loads videos (not yet implemented)
    """

    def __init__(self, root_dir, transform=None, is_frames=True):
        """Initialize the TrafficDataset.

        Args:
            root_dir: Path to directory containing images or videos
            transform: Optional torchvision transforms
            is_frames: Whether to load frames (True) or videos (False)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_frames = is_frames

        print(f"set is_frames to {is_frames}")

        frame_types = ('.jpg', '.png', '.jpeg')
        video_types = ('.mp4', '.avi', '.mov', '.gif')

        if self.is_frames:
            self.files = [f for f in os.listdir(self.root_dir)
                          if f.lower().endswith(frame_types)]
        else:
            self.files = [f for f in os.listdir(self.root_dir)
                          if f.lower().endswith(video_types)]

        # Sort files for consistency
        self.files.sort()

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            PIL.Image: Image in RGB format (for frames)

        Raises:
            NotImplementedError: If trying to load videos (not yet implemented)
        """
        if self.is_frames:
            img_path = self.root_dir / self.files[idx]
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        else:
            raise NotImplementedError("Video loading is not yet implemented")
