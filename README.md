# Vision Plus LLMs - Driver Assistance System

A modular, production-ready Python package for building vision-based driver assistance systems that combine object detection, motion analysis, and large language models.

## Overview

Vision Plus LLMs integrates:
- **Object Detection**: Uses DETR (Detection Transformer) to identify traffic objects (vehicles, pedestrians, traffic lights, stop signs, etc.)
- **Motion Analysis**: Computes centroids and velocities of detected objects across frames
- **Spatiotemporal Reasoning**: Tracks object motion over time to understand scene dynamics
- **LLM-Based Alerts**: Uses Flan-T5 or other LLMs to generate natural language alerts and warnings

## Project Structure

```
vision_plus_llms/
├── data/                 # Dataset loading (frames and videos)
├── models/               # Model loading utilities for DETR and LLMs
├── detection/            # Object detection with DETR
├── llm/                  # LLM integration (prompt construction, commentary)
├── video/                # Video processing with OpenCV
├── motion/               # Motion analysis and centroid computation
├── pipeline/             # End-to-end data pipeline
└── utils/                # Visualization and helper functions

examples/               # Example scripts
tests/                  # Test files
```

## Installation

### Using pip (after package setup)

```bash
pip install -e .
```

### Using pixi (recommended for reproducible environments)

Create a `pixi.toml` file in your workspace:

```toml
[project]
name = "vision_plus_llms"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[dependencies]
python = "3.11"
pytorch::pytorch-gpu = "*"
pytorch::pytorch = "*"
pytorch::torchvision = "*"
```

Then install:

```bash
pixi install
pip install -e .
```

## Quick Start

### Basic Object Detection

```python
import torch
from PIL import Image
from vision_plus_llms.models.model_loader import load_detection_model
from vision_plus_llms.detection.object_detector import ObjectDetector

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = load_detection_model(device=device)

# Create detector
detector = ObjectDetector(processor, model, device=device)

# Detect objects
image = Image.open("dashcam.jpg")
results = detector.detect(image, threshold=0.9)

for obj in results["objects"]:
    print(f"{obj['label']}: {obj['score']}")
```

### Full Pipeline with LLM Commentary

```python
import torch
from PIL import Image
from vision_plus_llms.models.model_loader import load_detection_model, load_llm_model
from vision_plus_llms.detection.object_detector import ObjectDetector
from vision_plus_llms.llm.prompt_constructor import PromptConstructor
from vision_plus_llms.llm.commentary_generator import CommentaryGenerator
from vision_plus_llms.pipeline.data_pipeline import DataPipeline

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_processor, detection_model = load_detection_model(device=device)
llm_tokenizer, llm_model = load_llm_model(device=device)

# Create components
detector = ObjectDetector(detection_processor, detection_model, device=device)
prompt_constructor = PromptConstructor()
commentary_generator = CommentaryGenerator(llm_tokenizer, llm_model, device=device)

# Create pipeline
pipeline = DataPipeline(detector, prompt_constructor, commentary_generator)

# Process frame
image = Image.open("dashcam.jpg")
result = pipeline.process_frame(image)

print("Detected objects:", [obj['label'] for obj in result['detections']])
print("LLM Alert:", result['commentary'])
```

### Motion Analysis Across Frames

```python
from vision_plus_llms.motion.motion_analyzer import MotionAnalyzer

# From consecutive detections
prev_objects = [{"label": "car", "box": [100, 100, 200, 200], "score": 0.95}]
curr_objects = [{"label": "car", "box": [110, 105, 210, 205], "score": 0.96}]

motion_data = MotionAnalyzer.compute_motion_from_objects(prev_objects, curr_objects)

for motion in motion_data:
    print(f"{motion['label']}: velocity = {motion['velocity']}, speed = {motion['speed']:.2f}")
```

## Module Usage

### Data Loading

```python
from vision_plus_llms.data import TrafficDataset

# Load frames from a directory
dataset = TrafficDataset("path/to/frames", is_frames=True)
image = dataset[0]
```

### Video Processing

```python
from vision_plus_llms.video import VideoReader

# Process video
with VideoReader("dashcam_video.mp4") as video:
    for frame_idx, frame in video.iter_frames(skip_frames=5):
        # Process frame...
```

### Visualization

```python
from vision_plus_llms.utils.visualization import draw_detections, draw_motion_vectors

# Visualize detections
draw_detections(image, detections, show=True)

# Visualize motion
draw_motion_vectors(image, motion_data, show=True)
```

## Example Scripts

### Run basic detection on an image

```bash
python examples/basic_detection.py path/to/image.jpg --threshold 0.9
```

### Run full pipeline on an image

```bash
python examples/full_pipeline.py path/to/image.jpg
```

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v --cov=vision_plus_llms
```

### Format code

```bash
black vision_plus_llms/ examples/ tests/
isort vision_plus_llms/ examples/ tests/
```

### Lint code

```bash
flake8 vision_plus_llms/ examples/ tests/
mypy vision_plus_llms/
```

## Extending the Project

### Adding a New Component

1. Create a new module folder in `vision_plus_llms/`
2. Implement your component(s)
3. Add `__init__.py` with exports
4. Update root `__init__.py` to import your components
5. Add tests in `tests/`
6. Add example usage in `examples/`

### Working with Team Members

Since this is a modular package:
- Each person can work on different modules independently
- Import components via `from vision_plus_llms import ComponentName`
- Use the examples as reference implementations
- Follow the existing code style and documentation patterns

## Key Classes and Functions

### `ObjectDetector`
Runs DETR-based object detection on images.

### `MotionAnalyzer`
Computes centroids, velocities, and acceleration from bounding boxes.

### `PromptConstructor`
Serializes object and motion data into LLM prompts.

### `CommentaryGenerator`
Generates natural language text from prompts using seq2seq models.

### `DataPipeline`
Orchestrates the end-to-end process from image to alert.

### `VideoReader`
Reads frames from video files efficiently with OpenCV.

## Dependencies

Core dependencies:
- `torch` >= 2.0.0
- `transformers` >= 4.25.0
- `opencv-python` >= 4.6.0
- `Pillow` >= 9.0.0
- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.0

See `pyproject.toml` for complete dependencies.

## License

MIT License

## References

- DETR Paper: https://arxiv.org/abs/2005.12368
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Flan-T5 Model: https://huggingface.co/models?search=flan-t5

## Contributing

Contributions are welcome! Please follow the project structure and documentation patterns.

## Contact

For questions or collaboration, reach out to the Digital Image Processing Group.
