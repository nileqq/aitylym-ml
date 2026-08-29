from transformers import AutoProcessor

BASE_MODEL = "facebook/hubert-base-ls960"

processor = AutoProcessor.from_pretrained(BASE_MODEL)