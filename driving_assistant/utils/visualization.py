"""Visualization utilities for object detection and motion analysis."""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from PIL import Image



class Visualizer:
    def __init__(self, output_path, codec='MP4V', fps=60, height=720, width=1280):
        self.output_path = output_path
        self.codec = codec
        self.fps = fps
        # Writer will be initialized lazily on first frame so we can infer
        # the true frame size from the incoming frames. Store intended
        # defaults in case no frame is provided.
        self._init_height = height
        self._init_width = width
        self.writer = None

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def init_writer(self, height, width, output_path):
        # Writer Initialization
        self.height = height
        self.width = width
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (width, height))
        if not getattr(self.writer, 'isOpened', lambda: True)():
            # Some OpenCV builds expose isOpened as a method on VideoWriter,
            # protect against older versions and signal failure early.
            raise RuntimeError(f"Failed to open VideoWriter for {output_path} using codec {self.codec}")

    def update(self, frame, detected_obj, metrics=None, commentary=None):
        """Processes the frame and writes it to the file."""
        annotated = self._draw_overlay(frame, detected_obj, metrics, commentary)

        # Lazily initialize writer with real frame size if needed
        if self.writer is None:
            h, w = annotated.shape[:2]
            try:
                self.init_writer(h, w, self.output_path)
            except Exception:
                # re-raise with more context
                raise

        # Ensure frame is uint8 and BGR
        if annotated.dtype != 'uint8':
            annotated = (np.clip(annotated, 0, 255)).astype('uint8')

        if self.writer is None:
            raise RuntimeError("VideoWriter not initialized")

        self.writer.write(annotated)

    def _draw_overlay(self, frame, detected_obj, metrics=None, commentary=None):
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

        # Overlay metrics panel
        if metrics:
            lines = [
                f"FPS: {metrics.get('fps', 0):.1f}",
                f"Objects: {metrics.get('obj_count', 0)}",
                f"OB Det: {metrics.get('avg_det_ms', 0):.0f}ms",
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

        # add commentary at the bottom if it exists
        if commentary:
            sub_h = int(self.height * 0.1)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, self.height - sub_h),
                          (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            cv2.putText(annotated, commentary, (30, self.height - int(sub_h/2.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated


def visualize_frame(frame, detected_obj, metrics=None):
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
