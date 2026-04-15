"""Traffic dataset loader for frames and videos."""

import os
from pathlib import Path
from PIL import Image
import torch


class TrafficDataset(torch.utils.data.Dataset):
    """Dataset for loading traffic-related frames and videos.

    This dataset supports loading individual frames from a directory. Video support
    is planned for future versions.

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
