import torch
import os
import argparse

try:
    from utils import load_model, MODEL_NAME_LIST
except ImportError:
    print("Error: Could not import from utils.py.")
    exit()

# Maps MODEL_NAME_LIST entries to their EOS token IDs
EOS_TOKEN_MAP = {
    'Llama-8B': 128001,
    'Qwen-1.5B': 151643,
    'Qwen-7B': 151643,
}


def main(data_id, attack_id, beam_size):
    # --- DYNAMIC PATH GENERATION ---
    task_name = f'attack_type:{attack_id}_model_type:{data_id}'
    latency_path = os.path.join('latency', f'{task_name}.latency')
    adv_path = os.path.join('adv', f'{task_name}_{beam_size}.adv')

    safe_task_name = task_name.replace(':', '_')
    report_path = f'{safe_task_name}_report.txt'

    if not os.path.exists(latency_path):
        print(f"Error: Latency file missing -> {latency_path}")
        return
    if not os.path.exists(adv_path):
        print(f"Error: Adv file missing -> {adv_path}")
        return

    latency_data = torch.load(latency_path)
    adv_data = torch.load(adv_path)

    # --- LOAD MODEL ---
    model_name = MODEL_NAME_LIST[data_id]

    if model_name not in EOS_TOKEN_MAP:
        print(f"Error: {model_name} is not a supported DeepSeek model for this report script.")
        print(f"Supported models: {list(EOS_TOKEN_MAP.keys())}")
        return

    eos_token_id = EOS_TOKEN_MAP[model_name]
    print(f"Loading {model_name} (EOS token id: {eos_token_id})...")
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    model.eval()

    print(f"Processing data... Saving report to {report_path}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 175}\n")
        f.write(f"FULL ADVERSARIAL GENERATION & REASONING REPORT\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Source: {latency_path}\n")
        f.write(f"{'=' * 175}\n")

        for i, (latency_entry, adv_history) in enumerate(zip(latency_data, adv_data)):
            metrics = latency_entry[0]

            f.write(f"\n[SENTENCE INDEX {i}]\n")
            f.write(f"{'-' * 175}\n")

            # Header
            f.write(
                f"{'Step':<20} | {'Latency (s)':<12} | {'Reasoning Tokens':<16} | {'Total Tokens(EOS)':<18} | {'Input Preview':<50} | {'Output Preview'}\n")
            f.write(f"{'-' * 175}\n")

            for step_idx, (lat_eng) in enumerate(metrics['cuda']):
                latency = lat_eng[0]
                step_label = "Original Input" if step_idx == 0 else f"Adversarial Input {step_idx}"

                full_text = adv_history[step_idx][0]

                # --- GENERATE LIVE RESPONSE ---
                inputs = tokenizer(full_text, return_tensors="pt", padding=True).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3000,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )

                    # Extract only newly generated tokens
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = outputs[0][input_length:].tolist()

                    # Strip at EOS and count total tokens up to EOS
                    if eos_token_id in generated_tokens:
                        eos_idx = generated_tokens.index(eos_token_id)
                        generated_tokens = generated_tokens[:eos_idx]

                    eos_token_count = len(generated_tokens)  # total tokens up to EOS

                    # Decode to find <think> tags
                    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

                    if "</think>" in full_generated_text:
                        parts = full_generated_text.split("</think>")
                        thinking_text = parts[0].replace("<think>", "").strip()
                        raw_output = parts[1].strip()
                        reasoning_count = len(tokenizer.encode(thinking_text, add_special_tokens=False))
                    else:
                        # model maxed out or didn't use tags
                        reasoning_count = len(generated_tokens)
                        eos_token_count = len(generated_tokens)
                        peek_text = tokenizer.decode(generated_tokens[-20:], skip_special_tokens=True).replace('\n', ' ')
                        raw_output = f"[NO TAGS / MAXED OUT] ...{peek_text}"

                    # --- FORMAT FOR TABLE ---
                    display_input = full_text.replace('\n', ' ')
                    display_output = raw_output.replace('\n', ' ')
                    if len(display_output) == 0:
                        display_output = "[Empty Output]"

                    f.write(
                        f"{step_label:<20} | {latency:<12.4f} | {reasoning_count:<16} | {eos_token_count:<18} | {display_input:<40} | {display_output}\n")

        f.write(f"\n{'=' * 175}\n")
        f.write(f"Total Sentences Processed: {len(latency_data)}\n")

    print(f"Done! Report saved successfully at: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and Format Latency Results')
    parser.add_argument('--data', required=True, type=int, help='model index from MODEL_NAME_LIST')
    parser.add_argument('--attack', required=True, type=int, help='attack type id')
    parser.add_argument('--beam', default=1, type=int, help='beam size (default: 1)')
    args = parser.parse_args()

    main(args.data, args.attack, args.beam)