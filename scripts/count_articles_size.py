import os
import argparse


def average_txt_length(root_folder):
    total_length_chars = 0
    total_length_words = 0
    file_count = 0

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                total_length_chars += len(content)
                total_length_words += len(content.split())
                file_count += 1

    if file_count == 0:
        return "No .txt files found."

    avg_chars = total_length_chars / file_count
    avg_words = total_length_words / file_count

    return {
        "number_of_files": file_count,
        "average_characters": avg_chars,
        "average_words": avg_words
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder with .txt files")
    args = parser.parse_args()

    result = average_txt_length(args.folder)
    print(result)
