import os
from pathlib import Path

def count_last_row_values(directory):
    """Count occurrences of 1s and 0s in the last row of each text file in the specified directory."""
    # Initialize counters
    count_1 = 0
    count_0 = 0

    # Create a Path object for the directory
    dir_path = Path(directory)

    # Loop through all text files in the directory
    for file in dir_path.glob("*.txt"):
        try:
            with open(file, 'r') as f:
                # Read all lines and get the last one
                lines = f.readlines()
                if lines:  # Check if the file is not empty
                    last_row = lines[-1].strip()  # Get the last row and remove any surrounding whitespace
                    # print(last_row)
                    if last_row == '1.0000000000000000e+00':
                        count_1 += 1
                    elif last_row == '0.0000000000000000e+00':
                        count_0 += 1
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    return count_1, count_0

if __name__ == "__main__":
    # Specify the directory you want to scan
    directory_to_scan = "./"  # Update this to your target directory

    # Count occurrences of 1s and 0s
    count_1, count_0 = count_last_row_values(directory_to_scan)

    # Print the results
    print(f"Total 1s found in last rows: {count_1}")
    print(f"Total 0s found in last rows: {count_0}")
