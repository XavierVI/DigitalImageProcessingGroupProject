"""Visualization utilities for object detection and motion analysis."""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from PIL import Image


def visualize_frame(frame, detected_obj, metrics=None):
    # Draw into a copy so we do not mutate the original frame.
    annotated = frame.copy()

    for obj in detected_obj:
        bx = [int(i) for i in obj["box"]]
        
        # Choose color based on detection type
        detection_type = obj.get('detection_type', 'vehicle')
        if detection_type == 'traffic_sign':
            box_color = (0, 165, 255)  # Orange for traffic signs
            text_color = (0, 165, 255)
        else:
            box_color = (255, 0, 0)  # Blue for vehicles
            text_color = (255, 0, 0)
        
        cv2.rectangle(
            annotated,
            (bx[0], bx[1]),
            (bx[2], bx[3]),
            box_color,
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

        # Format label with score and type
        label_text = f"{obj['label']} ({obj.get('score', 0):.2f})"
        cv2.putText(
            annotated,
            label_text,
            (bx[0], max(20, bx[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )

    # Overlay metrics panel
    if metrics:
        lines = [
            f"FPS: {metrics.get('fps', 0):.1f}",
            f"Objects: {metrics.get('obj_count', 0)}",
            f"Det: {metrics.get('avg_det_ms', 0):.0f}ms",
            f"LLM: {metrics.get('avg_llm_ms', 0):.0f}ms",
        ]
        pad = 6
        x, y = 10, 10
        line_h = 20
        panel_h = len(lines) * line_h + pad * 2
        panel_w = 180
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w,
                      y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (x + pad, y + pad + (i + 1) * line_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Alert System", annotated)
    cv2.waitKey(1)

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
