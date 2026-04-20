"""
This is the main script for running the project.


"""
import torch

from alert_system.data_stream.dataset import VideoDataset
from alert_system.pipeline.data_pipeline import DataPipeline

from transformers import DetrImageProcessor, DetrForObjectDetection, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def calculate_metrics(manual_labels, model_outputs):
    """
    manual_labels: dict { "vid_1": ["red_light", "night"] }
    model_outputs: dict { "vid_1": ["red_light", "intersection"] }

    The outputs can just be the words split up by spaces.
    """
    results = {}

    for vid, gt_tags in manual_labels.items():
        if vid not in model_outputs:
            continue

        pred_tags = set(model_outputs[vid])
        gt_tags = set(gt_tags)

        # Intersection over Union (Jaccard)
        intersection = gt_tags.intersection(pred_tags)
        union = gt_tags.union(pred_tags)
        jaccard = len(intersection) / len(union) if union else 0

        # Precision (How many predicted tags were right?)
        precision = len(intersection) / len(pred_tags) if pred_tags else 0

        # Recall (How many ground truth tags did we find?)
        recall = len(intersection) / len(gt_tags) if gt_tags else 0

        results[vid] = {
            "jaccard": round(jaccard, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3)
        }

    return results


def load_models(device):
    # loading models
    # Load Object Detection Model (DETR)
    model = "PekingU/rtdetr_r50vd"
    # model = "facebook/detr-resnet-50"

    ob_det_processor = RTDetrImageProcessor.from_pretrained(model)
    ob_det_model = RTDetrForObjectDetection.from_pretrained(model).to(device)
    ob_det_model.eval()

    # Load LLM for Commentary (Flan-T5-Small)
    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small").to(device)

    return ob_det_processor, ob_det_model, llm_tokenizer, llm_model


device = "cuda" if torch.cuda.is_available() else "cpu"
reddit_videos = './data/reddit_dashcam_videos/'
dataset = VideoDataset(root_dir=reddit_videos)
# TODO: load the labels for evaluation.

# open a video
dataset[5]

ob_det_processor, ob_det_model, llm_tokenizer, llm_model = load_models(device=device)

print("Models loaded successfully. Starting pipeline...")

pipeline = DataPipeline(
    image_processor=ob_det_processor,
    object_detection_model=ob_det_model,
    llm_tokenizer=llm_tokenizer,
    llm_model=llm_model,
    datastream=dataset,
    device=device,
    window_size=10
)

# try doing one object detection step
# success, frame = dataset.step()
# print(type(frame))
# print(frame.shape)
# print(pipeline._obj_detection(frame))

# TODO: in the future, we are going to wrap this code in a loop
llm_output = pipeline.loop(visualize=True)

# turn output into dict { "vid_1": ["red_light", "night"] }
# for simplicity, we just split all the commentary into words
model_outputs = {}
full_output = []

for timestep, commentary in llm_output:
    tags = commentary.split()
    full_output += tags

model_outputs[dataset.get_video_name(0)] = full_output