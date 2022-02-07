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

encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "distilgpt2"
model_checkpoint = "gagan3012/ViTGPT2_vizwiz"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)
tts_models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
)

def predict_image_to_text(image_path):
    image = Image.open(image_path).convert("RGB")
    def clean_text(x): return x.replace("<|endoftext|>", "").split("\n")[0]
    sample = feature_extractor(
        image, return_tensors="pt").pixel_values.to(device)
    caption_ids = model.generate(sample, max_length=50)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))
    return caption_text


def predict_text_to_audio(text):
    model = tts_models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg,task.data_cfg)
    generator = task.build_generator(model, cfg)

    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav, rate

def image_to_audio(image_path):
    caption_text = predict_image_to_text(image_path)
