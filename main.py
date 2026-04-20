"""
This is the main script for running the project.


"""
import torch

from alert_system.data_stream.dataset import VideoDataset
from alert_system.pipeline.data_pipeline import DataPipeline

from transformers import DetrImageProcessor, DetrForObjectDetection, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


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

# open a video
dataset[7]

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

pipeline.loop(visualize=False)
