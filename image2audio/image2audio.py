import torch
from PIL import Image
from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

