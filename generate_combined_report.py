import torch
import os
import time
import argparse
from tqdm import tqdm
import pynvml

try:
    from utils import load_model, MODEL_NAME_LIST
except ImportError:
    print("Error: Could not import from utils.py.")
    exit()

# ── Constants ──────────────────────────────────────────────────────────────────
EOS_TOKEN_MAP = {
    'Llama-8B': 128001,
    'Qwen-1.5B': 151643,
    'Qwen-7B': 151643,
}

MAX_NUM  = 100   # max sentences to process
MAX_PER  = 3     # max adversarial steps per sentence
ITER_NUM = 3     # GPU measurement iterations


# ── Core: single generation that yields latency + reasoning tokens ─────────────
@torch.no_grad()
def generate_and_measure(model, tokenizer, text, eos_token_id, device):
    """
    One generation call that simultaneously:
      - measures GPU latency / energy
      - counts reasoning tokens inside <think>…</think>
      - counts total tokens up to EOS
      - returns the decoded output preview
    Returns dict with keys: latency, energy, reasoning_tokens,
                             total_tokens, output_preview
    """
    inputs     = tokenizer(text, return_tensors="pt", padding=True).to(device)
    input_len  = inputs.input_ids.shape[1]

    # ── latency measurement ────────────────────────────────────────────────────
    pynvml.nvmlInit()
    dev_id = 0 if device.index is None else device.index
    handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)

    t1, power_list = time.time(), []
    for _ in range(ITER_NUM):
        outputs = model.generate(
            **inputs,
            max_new_tokens=3000,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        power_list.append(pynvml.nvmlDeviceGetPowerUsage(handle))
    t2 = time.time()
    pynvml.nvmlShutdown()

    latency = (t2 - t1) / ITER_NUM
    energy  = (sum(power_list) / len(power_list) * (t2 - t1)) / (10 ** 6) / ITER_NUM

    # ── decode & parse reasoning tokens (use last generation) ─────────────────
    generated_tokens = outputs[0][input_len:].tolist()

    if eos_token_id in generated_tokens:
        generated_tokens = generated_tokens[:generated_tokens.index(eos_token_id)]

    total_tokens      = len(generated_tokens)
    full_decoded      = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    if "</think>" in full_decoded:
        thinking_text    = full_decoded.split("</think>")[0].replace("<think>", "").strip()
        raw_output       = full_decoded.split("</think>")[1].strip()
        reasoning_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))
    else:
        reasoning_tokens = total_tokens
        peek             = tokenizer.decode(generated_tokens[-20:], skip_special_tokens=True)
        raw_output       = f"[NO TAGS / MAXED OUT] ...{peek}"

    return {
        "latency":          latency,
        "energy":           energy,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens":     total_tokens,
        "output_preview":   raw_output.replace("\n", " "),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_id, attack_id, beam_size):
    model_name = MODEL_NAME_LIST[data_id]

    if model_name not in EOS_TOKEN_MAP:
        print(f"Error: '{model_name}' is not a supported model.")
        print(f"Supported: {list(EOS_TOKEN_MAP.keys())}")
        return

    eos_token_id = EOS_TOKEN_MAP[model_name]
    task_name    = f'attack_type:{attack_id}_model_type:{data_id}'
    adv_path     = os.path.join('adv', f'{task_name}_{beam_size}.adv')
    report_path  = f'{task_name.replace(":", "_")}_report.txt'

    if not os.path.exists(adv_path):
        print(f"Error: Adversarial file not found -> {adv_path}")
        return

    print(f"Loading {model_name} (EOS token id: {eos_token_id})...")
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    model.eval()
    device = next(model.parameters()).device

    adv_res = torch.load(adv_path)

    print(f"Processing {min(len(adv_res), MAX_NUM)} sentences...")
    print(f"Report -> {report_path}")

    SEP = "=" * 175
    COL = "-" * 175

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{SEP}\n")
        f.write(f"COMBINED LATENCY & REASONING TOKEN REPORT\n")
        f.write(f"Model : {model_name}\n")
        f.write(f"Source: {adv_path}\n")
        f.write(f"{SEP}\n")

        header = (f"{'Step':<22} | {'Latency (s)':<12} | {'Energy (J)':<11} | "
                  f"{'Reasoning Tok':<15} | {'Total Tok':<10} | "
                  f"{'Input Preview':<45} | Output Preview")
        f.write(f"\n{header}\n{COL}\n")

        for i, adv_his in tqdm(enumerate(adv_res), total=min(len(adv_res), MAX_NUM)):
            if i >= MAX_NUM:
                break

            adv_his = adv_his[:MAX_PER + 1]

            f.write(f"\n[SENTENCE INDEX {i}]\n{COL}\n")

            prev_text = None
            for step_idx, (adv_text, orig_len, overheads) in enumerate(adv_his):
                step_label = "Original Input" if step_idx == 0 else f"Adversarial {step_idx}"

                # ── skip duplicate inputs ──────────────────────────────────────
                if prev_text == adv_text:
                    f.write(f"{step_label:<22} | [duplicate – skipped]\n")
                    continue

                prev_text = adv_text

                # ── single call: latency + reasoning tokens ────────────────────
                result = generate_and_measure(
                    model, tokenizer, adv_text, eos_token_id, device
                )

                # ── write report row ───────────────────────────────────────────
                display_in  = adv_text.replace("\n", " ")
                display_out = result["output_preview"]

                f.write(
                    f"{step_label:<22} | "
                    f"{result['latency']:<12.4f} | "
                    f"{result['energy']:<11.4f} | "
                    f"{result['reasoning_tokens']:<15} | "
                    f"{result['total_tokens']:<10} | "
                    f"{display_in:<45} | "
                    f"{display_out}\n"
                )

        f.write(f"\n{SEP}\n")
        f.write(f"Total Sentences Processed: {min(len(adv_res), MAX_NUM)}\n")

    print(f"\nDone! Report saved -> {report_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure latency and generate reasoning report in one pass.")
    parser.add_argument("--data",   required=True, type=int, help="Model index from MODEL_NAME_LIST")
    parser.add_argument("--attack", required=True, type=int, help="Attack type ID")
    parser.add_argument("--beam",   default=1,     type=int, help="Beam size (default: 1)")
    args = parser.parse_args()

    main(args.data, args.attack, args.beam)