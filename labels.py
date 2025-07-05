import yaml
import os

def get_all_unique_characters(label_paths, config_path="config.yaml"):
    all_chars = set()
    missing_chars_in_charset = set()

    # Load charset from config.yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        current_charset = set(config["dataset"]["charset"])
        print(f"Current charset from config.yaml: '{config['dataset']['charset']}'")
        print(f"Length of current charset: {len(current_charset)}")
    except FileNotFoundError:
        print(f"Error: config.yaml not found at {config_path}")
        return

    for label_path in label_paths:
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Assuming format: image_path\tlabel
                    _, label = line.strip().split('\t', 1)
                    for char in label:
                        all_chars.add(char)
                        if char not in current_charset:
                            missing_chars_in_charset.add(char)
                except ValueError:
                    print(f"Warning: Skipping malformed line {line_num} in {label_path}: '{line.strip()}'")
                except Exception as e:
                    print(f"Error processing line {line_num} in {label_path}: {e} - Line: '{line.strip()}'")

    print("\n--- Analysis Results ---")
    
    if all_chars:
        sorted_all_chars = sorted(list(all_chars))
        print(f"\nAll unique characters found in your label files (sorted):")
        print(f"'{''.join(sorted_all_chars)}'")
        print(f"Total unique characters found: {len(sorted_all_chars)}")
    else:
        print("No characters found in labels. Ensure label files are correctly populated.")
        return

    if missing_chars_in_charset:
        sorted_missing_chars = sorted(list(missing_chars_in_charset))
        print(f"\nCharacters found in labels but MISSING from your current charset:")
        print(f"'{''.join(sorted_missing_chars)}'")
        print("You MUST add these characters to your 'charset' in config.yaml.")
    else:
        print("\nAll characters in your label files are present in your current charset. This is good!")

if __name__ == "__main__":
    # Define the paths to your label files
    train_labels_path = "./dataset/train_labels.txt" #
    val_labels_path = "./dataset/val_labels.txt" #
    
    get_all_unique_characters([train_labels_path, val_labels_path])