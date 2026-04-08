import torch
import os
import argparse


def main(data_id, attack_id, beam_size):
    # --- DYNAMIC PATH GENERATION ---
    task_name = f'attack_type:{attack_id}_model_type:{data_id}'

    latency_path = os.path.join('latency', f'{task_name}.latency')
    adv_path = os.path.join('adv', f'{task_name}_{beam_size}.adv')

    # --- NEW: Output Report File Path ---
    # Convert the colons to underscores so the file can be downloaded safely!
    safe_task_name = task_name.replace(':', '_')
    report_path = f'{safe_task_name}_report.txt'

    # Check for missing files explicitly so you know WHICH one is missing
    if not os.path.exists(latency_path):
        print(f"Error: Latency file missing -> {latency_path}")
        return
    if not os.path.exists(adv_path):
        print(f"Error: Adv file missing -> {adv_path}")
        return

    # Load both datasets
    latency_data = torch.load(latency_path)
    adv_data = torch.load(adv_path)

    print(f"Processing data... Saving report to {report_path}")

    # Open the text file in write mode ('w')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 95}\n")
        f.write(f"FULL ADVERSARIAL LATENCY REPORT\n")
        f.write(f"Source: {latency_path}\n")
        f.write(f"{'=' * 95}\n")

        # Iterate through the sentences
        for i, (latency_entry, adv_history) in enumerate(zip(latency_data, adv_data)):
            # latency_entry is (metric_dict, original_flops)
            metrics = latency_entry[0]

            f.write(f"\n[SENTENCE INDEX {i}]\n")
            f.write(f"{'-' * 95}\n")

            # Header for the table (Widened the Step column to 20 spaces)
            f.write(f"{'Step':<20} | {'Latency (s)':<12} | {'Energy (mJ)':<12} | {'Text Preview'}\n")
            f.write(f"{'-' * 95}\n")

            # Iterate through the steps of the attack for this sentence
            # We use 'cuda' metrics specifically
            for step_idx, (lat_eng) in enumerate(metrics['cuda']):
                latency = lat_eng[0]
                energy = lat_eng[1]

                # --- NEW: Create the descriptive label ---
                if step_idx == 0:
                    step_label = "Original Input"
                else:
                    step_label = f"Adversarial Input {step_idx}"

                # Get the text from the .adv file (first element of the tuple)
                # adv_history[step_idx] is (text, length, overhead)
                full_text = adv_history[step_idx][0]

                # Clean up text for printing (remove newlines)
                display_text = full_text.replace('\n', ' ')

                # Write using the new label and widened column
                f.write(f"{step_label:<20} | {latency:<12.4f} | {energy:<12.4f} | {display_text}\n")

        f.write(f"\n{'=' * 95}\n")
        f.write(f"Total Sentences Processed: {len(latency_data)}\n")

    print(f"Done! Report saved successfully at: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and Format Latency Results')
    parser.add_argument('--data', required=True, type=int, help='experiment subjects (model_type id)')
    parser.add_argument('--attack', required=True, type=int, help='attack type id')
    parser.add_argument('--beam', default=1, type=int, help='beam size (default: 1)')
    args = parser.parse_args()

    main(args.data, args.attack, args.beam)