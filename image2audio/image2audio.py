import torch
from PIL import Image
from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                          ViTFeatureExtractor)

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
