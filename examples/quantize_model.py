import torch
import numpy as np

from PIL import Image
import gc
import requests
from io import BytesIO

from transformers import AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import real_quantize_model_weight

from tinychat.utils.load_quant import load_awq_model
from tinychat.utils.tune import device_warmup, tune_all_wqlinears
from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
from tinychat.utils.llava_image_processing import process_images, load_image
from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp, make_fused_vision_attn

import os

# 8 GPU devices are used
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

model_path = "/models/gemma-7b"
quant_path = ""