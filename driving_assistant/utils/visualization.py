"""Visualization utilities for object detection and motion analysis."""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from PIL import Image


def visualize_frame(self, frame, detected_obj):
    # Draw into a copy so we do not mutate the original frame.
    annotated = frame.copy()

    for obj in detected_obj:
        bx = [int(i) for i in obj["box"]]
        cv2.rectangle(
            annotated,
            (bx[0], bx[1]),
            (bx[2], bx[3]),
            (255, 0, 0),
            2,
        )

        c = tuple(map(int, obj["centroid"]))
        m = obj.get("velocity", (0, 0))
        end_point = (int(c[0] + m[0]), int(c[1] + m[1]))
        if end_point != c:
            cv2.arrowedLine(
                annotated,
                c,
                end_point,
                (0, 255, 0),
                2,
                tipLength=0.3,
            )

        cv2.putText(
            annotated,
            str(obj["label"]),
            (bx[0], max(20, bx[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Fast real-time rendering for video playback.
    cv2.imshow("Alert System", annotated)
    cv2.waitKey(1)



def draw_detections(
    image: Image.Image,
    detected_objects: List[Dict],
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True
) -> None:
    """Draw detected objects on an image.

    Args:
        image (PIL.Image): Input image
        detected_objects (list): List of detected objects with labels, scores, boxes
        figsize (tuple): Figure size
        show (bool): Whether to display the plot
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for idx, obj in enumerate(detected_objects):
        xmin, ymin, xmax, ymax = obj['box']
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"{obj['label']} ({idx}): {obj['score']:.2f}"
        ax.text(
            xmin, ymin - 10,
            label_text,
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10,
            color='white'
        )

    ax.axis('off')
    plt.title("Object Detections")

    if show:
        plt.show()


def draw_motion_vectors(
    image: Image.Image,
    motion_data: List[Dict],
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True
) -> None:
    """Draw motion vectors for detected objects.

    Args:
        image (PIL.Image): Input image
        motion_data (list): List of motion data with centroids and velocities
        figsize (tuple): Figure size
        show (bool): Whether to display the plot
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for motion in motion_data:
        if not motion.get('matched', False):
            continue

        prev_centroid = motion['previous_centroid']
        curr_centroid = motion['current_centroid']

        # Draw arrow from previous to current centroid
        ax.annotate(
            '',
            xy=curr_centroid,
            xytext=prev_centroid,
            arrowprops=dict(arrowstyle='->', color='lime', lw=2)
        )

        # Draw current position
        ax.plot(curr_centroid[0], curr_centroid[1], 'go', markersize=8)

        # Add label with velocity
        speed = motion['speed']
        ax.text(
            curr_centroid[0] + 10,
            curr_centroid[1] + 10,
            f"{motion['label']}\nv={speed:.1f}",
            bbox=dict(facecolor='green', alpha=0.5),
            fontsize=9,
            color='white'
        )

    ax.axis('off')
    plt.title("Object Motion Vectors")

    if show:
        plt.show()


def draw_coordinate_frame(
    image: Image.Image,
    origin: Tuple[float, float] = (0, 0),
    axis_length: int = 100,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True
) -> None:
    """Draw a coordinate frame on an image (useful for camera calibration visualization).

    Args:
        image (PIL.Image): Input image
        origin (tuple): (u, v) pixel coordinates of frame origin
        axis_length (int): Length of axis lines in pixels
        figsize (tuple): Figure size
        show (bool): Whether to display the plot
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    u, v = origin
    # X-axis (red)
    ax.annotate(
        '',
        xy=(u + axis_length, v),
        xytext=(u, v),
        arrowprops=dict(arrowstyle='->', color='red', lw=3)
    )
    # Y-axis (green)
    ax.annotate(
        '',
        xy=(u, v + axis_length),
        xytext=(u, v),
        arrowprops=dict(arrowstyle='->', color='green', lw=3)
    )

    ax.text(u + axis_length + 10, v, 'X', fontsize=12, color='red', weight='bold')
    ax.text(u, v + axis_length + 10, 'Y', fontsize=12, color='green', weight='bold')

    ax.axis('off')
    plt.title("Coordinate Frame")

    if show:
        plt.show()
