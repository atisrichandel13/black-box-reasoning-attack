import torch
import os
import argparse

# Import your model loader
try:
    from generate_adv import load_model, MODEL_NAME_LIST
except ImportError:
    print("Error: Could not import load_model from generate_adv.py.")
    exit()


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

    # --- 1. BOOT UP THE MODEL ---
    model_name = MODEL_NAME_LIST[data_id]
    print(f"Loading {model_name}...")
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    model.eval()

    print(f"Processing data... Saving report to {report_path}")

    # The exact Llama 3.1 End-of-Turn / End-of-Reasoning Token ID
    EOT_TOKEN_ID = 128009

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 155}\n")
        f.write(f"FULL ADVERSARIAL GENERATION & REASONING REPORT\n")
        f.write(f"Source: {latency_path}\n")
        f.write(f"{'=' * 155}\n")

        for i, (latency_entry, adv_history) in enumerate(zip(latency_data, adv_data)):
            metrics = latency_entry[0]

            f.write(f"\n[SENTENCE INDEX {i}]\n")
            f.write(f"{'-' * 155}\n")

            # Header
            f.write(
                f"{'Step':<20} | {'Latency (s)':<12} | {'Reasoning Tokens':<16} | {'Input Preview':<50} | {'Output Preview'}\n")
            f.write(f"{'-' * 155}\n")

            for step_idx, (lat_eng) in enumerate(metrics['cuda']):
                latency = lat_eng[0]
                step_label = "Original Input" if step_idx == 0 else f"Adversarial Input {step_idx}"

                full_text = adv_history[step_idx][0]

                # --- 2. GENERATE LIVE RESPONSE ---
                inputs = tokenizer(full_text, return_tensors="pt", padding=True).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3000,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )

                    # Extract only the newly generated tokens as a standard Python list
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = outputs[0][input_length:].tolist()

                    # --- BULLETPROOF REASONING SPLIT FOR BASE MODEL ---
                    # 1. Check for the absolute end of the generation (Base model uses 128001)
                    if 128001 in generated_tokens:
                        eos_idx = generated_tokens.index(128001)
                        generated_tokens = generated_tokens[:eos_idx]

                    # 2. Decode to text so we can find the literal tags
                    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

                    if "</think>" in full_generated_text:
                        # Split the text perfectly in half at the closing tag
                        parts = full_generated_text.split("</think>")
                        thinking_text = parts[0].replace("<think>", "").strip()
                        raw_output = parts[1].strip()

                        # Count the reasoning tokens by re-encoding just the thinking part
                        reasoning_count = len(tokenizer.encode(thinking_text, add_special_tokens=False))

                    else:
                        # The model didn't use the tags, or hit the max token limit before finishing
                        reasoning_count = len(generated_tokens)
                        peek_text = tokenizer.decode(generated_tokens[-20:], skip_special_tokens=True).replace('\n',
                                                                                                               ' ')
                        raw_output = f"[NO TAGS / MAXED OUT] ...{peek_text}"
                    # --- 4. FORMAT FOR TABLE ---
                    display_input = full_text.replace('\n', ' ')
                    if len(display_input) > 37:
                        display_input = display_input

                    display_output = raw_output.replace('\n', ' ')
                    if len(display_output) > 55:
                        display_output = display_output
                    elif len(display_output) == 0:
                        display_output = "[Empty Output]"

                    f.write(
                        f"{step_label:<20} | {latency:<12.4f} | {reasoning_count:<16} | {display_input:<40} | {display_output}\n")
        f.write(f"\n{'=' * 155}\n")
        f.write(f"Total Sentences Processed: {len(latency_data)}\n")

    print(f"Done! Report saved successfully at: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and Format Latency Results')
    parser.add_argument('--data', required=True, type=int, help='experiment subjects (model_type id)')
    parser.add_argument('--attack', required=True, type=int, help='attack type id')
    parser.add_argument('--beam', default=1, type=int, help='beam size (default: 1)')
    args = parser.parse_args()

    main(args.data, args.attack, args.beam)