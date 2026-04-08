import torch
import os
import argparse

# FIXED: import from utils directly, not from generate_adv
from utils import load_model, MODEL_NAME_LIST


def main(data_id, attack_id, beam_size):
    task_name = f'attack_type:{attack_id}_model_type:{data_id}'
    latency_path = os.path.join('latency', f'{task_name}.latency')
    adv_path = os.path.join('adv', f'{task_name}_{beam_size}.adv')

    safe_task_name = task_name.replace(':', '_')
    report_path = f'{safe_task_name}_input_output_report.txt'

    if not os.path.exists(latency_path):
        print(f"Error: Latency file missing -> {latency_path}")
        return
    if not os.path.exists(adv_path):
        print(f"Error: Adv file missing -> {adv_path}")
        return

    latency_data = torch.load(latency_path)
    adv_data = torch.load(adv_path)

    model_name = MODEL_NAME_LIST[data_id]
    print(f"Loading {model_name} from utils.py...")

    # FIXED: load_model returns 5 values (no dataset)
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    model.eval()

    print(f"Processing data... Saving report to {report_path}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 135}\n")
        f.write(f"FULL ADVERSARIAL GENERATION REPORT\n")
        f.write(f"Source: {latency_path}\n")
        f.write(f"{'=' * 135}\n")

        for i, (latency_entry, adv_history) in enumerate(zip(latency_data, adv_data)):
            metrics = latency_entry[0]

            f.write(f"\n[SENTENCE INDEX {i}]\n")
            f.write(f"{'-' * 135}\n")
            f.write(f"{'Step':<20} | {'Latency (s)':<12} | {'Input Preview':<40} | {'Output Preview'}\n")
            f.write(f"{'-' * 135}\n")

            for step_idx, (lat_eng) in enumerate(metrics['cuda']):
                latency = lat_eng[0]
                step_label = "Original Input" if step_idx == 0 else f"Adversarial Input {step_idx}"

                full_text = adv_history[step_idx][0]

                inputs = tokenizer(full_text, return_tensors="pt", padding=True).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )

                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                display_input = full_text.replace('\n', ' ')
                if len(display_input) > 37:
                    display_input = display_input

                display_output = raw_output.replace('\n', ' ')
                if len(display_output) > 55:
                    display_output = display_output
                elif len(display_output) == 0:
                    display_output = "[Empty Output]"

                f.write(f"{step_label:<20} | {latency:<12.4f} | {display_input:<40} | {display_output}\n")

        f.write(f"\n{'=' * 135}\n")
        f.write(f"Total Sentences Processed: {len(latency_data)}\n")

    print(f"Done! Report saved successfully at: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and Format Latency Results')
    parser.add_argument('--data', required=True, type=int, help='experiment subjects (model_type id)')
    parser.add_argument('--attack', required=True, type=int, help='attack type id')
    parser.add_argument('--beam', default=1, type=int, help='beam size (default: 1)')
    args = parser.parse_args()

    main(args.data, args.attack, args.beam)