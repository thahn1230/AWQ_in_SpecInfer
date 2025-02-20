import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon

from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm


# awq 진행
model_path = 'facebook/opt-6.7b'
w_bit = 8 # LTM weight bit width
load_awq = './opt-6.7b-w8-g128.pt' # save quantized model

batch_size = 1
tasks = None # wikitext for evaluation
output_path = None # if no need to save the model, put None
num_fewshot = 0
q_group_size = 128 # 128 is default in research paper
q_backend = "fake" # pseudo quantization
dump_fake = None # save fake-quantized model
dump_awq = None
load_quant = None # load quantized model
run_awq = False
dump_quant = None


smooth_scale = False
no_zero_point = False
vila_15 = False
vila_20 = False
max_memory = {}

ssm_vila_10_quant_mode = (
    ("llava" in model_path.lower() or "vila" in model_path.lower())
    and not vila_15
    and not vila_20
)

max_memory = [v.split(":") for v in (max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not no_zero_point,  # by default True
    "q_group_size": q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

def build_model_and_enc(model_path, w_bit, load_awq):
    if not os.path.exists(model_path):  # 로컬 경로가 아니면 Hugging Face 모델로 가정
        print(f"Model path {model_path} not found locally. Attempting to download from Hugging Face.")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True
        )
    except Exception as e:
            raise RuntimeError(f"Failed to download or load the Hugging Face model: {e}")
    else:
        print(f"* Building model {model_path}")

    # all hf model
    if ssm_vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        run_awq = True

        run_awq &= not load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not ssm_vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if run_awq:
            assert dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if dump_awq:
                dirpath = os.path.dirname(dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, dump_awq)
                print("AWQ results saved at", dump_awq)

            exit(0)

        if load_awq:
            print("Loading pre-computed AWQ results from", load_awq)
            awq_results = torch.load(load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if w_bit is not None:
            if q_backend == "fake":
                dump_quant = None
                assert (
                    dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
                if dump_fake:
                    model.save_pretrained(dump_fake)
                    print("Pseudo-quantized models saved at", dump_fake)
            elif q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
                if dump_quant:
                    if not dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        dump_quant = dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {dump_quant}...")
                    torch.save(model.cpu().state_dict(), dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, tokenizer, enc

if output_path is not None and os.path.exists(output_path):
    # print(f"Results {args.output_path} already generated. Exit.")
    print(f"Results {output_path} already generated. Overwrite.")
    # exit()

# a hack here to auto set model group
if smooth_scale and vila_20:
    if os.path.exists(act_scale_path):
        print(f"Found existing Smooth Scales {act_scale_path}, skip.")
    else:
        from awq.quantize import get_smooth_scale

        act_scale = get_smooth_scale(model_path, media_path)
        os.makedirs(os.path.dirname(act_scale_path), exist_ok=True)
        torch.save(act_scale, act_scale_path)
        print("Save act scales at " + str(act_scale_path))
        model_path = model_path + "/llm"
    if dump_awq is None and dump_quant is None:
        exit()

if dump_awq and os.path.exists(dump_awq):
    print(f"Found existing AWQ results {dump_awq}, exit.")
    exit()

model, tokenizer, enc = build_model_and_enc(model_path, w_bit, load_awq)


if tasks is not None:
    # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
    if tasks == "wikitext":
        testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
        model.seqlen = 2048
        testenc = testenc.input_ids.to(model.device)
        nsamples = testenc.numel() // model.seqlen
        model = model.eval()
        nlls = []
        for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                model.device
            )
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[
                :, (i * model.seqlen) : ((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(ppl.item())

        results = {"ppl": ppl.item()}
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
    else:
        task_names = tasks.split(",")

        lm_eval_model = LMEvalAdaptor(model_path, model, enc, batch_size)
        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=batch_size,
            no_cache=True,
            num_fewshot=num_fewshot,
        )

        print(evaluator.make_table(results))

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = model_path
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)




# 모델 및 토크나이저 로드
model2_name = "facebook/opt-1.3b"
tokenizer1 = tokenizer
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model1 = model
model2 = AutoModelForCausalLM.from_pretrained(model2_name)

# 데이터셋 로드 및 입력 프롬프트 선택
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
prompt = dataset[0]["text"]

# 토크나이징
inputs1 = tokenizer1(prompt, return_tensors="pt")
inputs2 = tokenizer2(prompt, return_tensors="pt")

# 모델 예측 (logits 계산)
with torch.no_grad():
    logits1 = model1(**inputs1).logits
    logits2 = model2(**inputs2).logits

# 확률 계산 (softmax)
probs1 = torch.softmax(logits1[0, -1], dim=-1).cpu().numpy()
probs2 = torch.softmax(logits2[0, -1], dim=-1).cpu().numpy()

# JS Divergence 계산
def calculate_js_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return jensenshannon(p, q) ** 2

js_divergence = calculate_js_divergence(probs1, probs2)

# 토큰 분포 그래프 그리기
vocab_size = len(probs1)
tokens = np.arange(vocab_size)

plt.figure(figsize=(10, 6))
plt.plot(tokens, probs1, label=f"opt-6.7b 8-bit Distribution", alpha=0.7)
plt.plot(tokens, probs2, label=f"{model2_name} Distribution", alpha=0.7)
plt.title(f"Token Distribution Comparison\n(JS Divergence: {js_divergence:.4f})")
plt.xlabel("Token Index")
plt.ylabel("Probability")
plt.legend()
plt.yscale("log")  # 로그 스케일로 보기 좋게
plt.grid(True)

# 그래프 저장
plt.savefig("token_distribution_comparison.png")
plt.show()
