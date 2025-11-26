import re
import sys
import os

def find_best_accuracy(log_content):
    """
    Parses the log content to find the best Top1 validation accuracy and its corresponding epoch.
    """
    best_accuracy = -1.0
    best_epoch = -1
    current_epoch = -1

    # Regex for extracting the epoch number from evaluation logs
    epoch_re = re.compile(r"Eval epoch: (\d+)")
    # Regex for extracting the Top1 accuracy
    top1_re = re.compile(r"Top1: ([\d.]+)%")

    for line in log_content.splitlines():
        # 1. Check for the current evaluation epoch
        epoch_match = epoch_re.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        # 2. Check for Top1 accuracy
        top1_match = top1_re.search(line)
        if top1_match:
            try:
                # Convert the accuracy string to a float
                accuracy = float(top1_match.group(1))
                # 3. Update best accuracy if the current one is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = current_epoch
            except ValueError:
                # Skip if conversion fails (e.g., malformed percentage)
                pass

    return best_accuracy, best_epoch

def main():
    # Check if a file path argument was passed
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_log_file>")
        sys.exit(1)

    log_file_path = sys.argv[1]

    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at '{log_file_path}'")
        sys.exit(1)

    try:
        # Read the entire log file content
        with open(log_file_path, 'r') as f:
            log_content = f.read()

    except Exception as e:
        print(f"Error reading file '{log_file_path}': {e}")
        sys.exit(1)

    # Process the log content
    best_accuracy, best_epoch = find_best_accuracy(log_content)

    if best_epoch != -1:
        print(f"\n✅ Results for log file: {log_file_path}")
        print("-" * 40)
        print(f"Best Validation Accuracy (Top1): {best_accuracy}%")
        print(f"Corresponding Epoch: {best_epoch}")
    else:
        print(f"\n⚠️ Could not find any Top1 validation accuracy data in the log file: {log_file_path}")

if __name__ == "__main__":
    main()