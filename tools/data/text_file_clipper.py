import os
import random
import argparse

def create_random_text_file(input_file: str, size_mb: float, output_file: str):
    # Read the entire input file
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text:
        raise ValueError("Input file is empty!")

    target_size_bytes = int(size_mb * 1024 * 1024)
    generated_text = []

    while sum(len(chunk.encode("utf-8")) for chunk in generated_text) < target_size_bytes:
        # Pick a random start index
        start = random.randint(0, max(0, len(text) - 1000))
        end = start + random.randint(200, 1000)  # pick a random chunk
        generated_text.append(text[start:end])

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(generated_text))

    print(f"✅ Created {output_file} with ~{size_mb} MB of random text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random text from input file to desired size in MB.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("size_mb", type=float, help="Desired output size in MB (e.g. 0.2)")
    parser.add_argument("output_file", help="Path to save the output file")

    args = parser.parse_args()
    create_random_text_file(args.input_file, args.size_mb, args.output_file)
