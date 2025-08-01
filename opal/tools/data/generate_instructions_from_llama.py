import re
import os
import json
import requests
import argparse
import logging
from pathlib import Path

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def chunk_text(text, max_tokens=1000):
    """Split text into chunks of ~max_tokens words."""
    words = text.split()
    chunks, current_chunk = [], []
    count = 0

    for word in words:
        current_chunk.append(word)
        count += 1
        if count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, count = [], 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def generate_instructions(chunk, model="dolphin3", num_instructions=7):
    """Send chunk to Ollama REST API to generate instructions."""

    logging.info(f"Sending chunk of size {len(chunk.split())} words to model {model}...")

    prompt = f"""
    You are an expert Cisco network engineer and technical writer.
    Generate {num_instructions} diverse high-quality pretraining instruction examples.

    Each instruction must be a **single JSON object** containing:
    - "prompt": The natural language user request (how a user might ask Olaf)
    - "action": One of [orbit.teach, orbit.configure, orbit.show, orbit.explain, orbit.troubleshoot, orbit.summary]
    - Note: Make sure orbit.configure must have only the commands and not the steps. The commands usually prefixed with router/switch(config*# 
        otherwise change action to orbit.explain
    - Note: Make sure orbit.show must have only the commands and not the steps.
    - "description": A short description of what the instruction accomplishes
    - "response": Either a **JSON object with steps or commands** OR a **single string** with concise commands

    Example formats:
    ...
    """

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024}
    }

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=data, timeout=300)
        resp.raise_for_status()
        output_text = resp.json().get("response", "")
        logging.info(f"Received response of length {len(output_text)} characters.")
    except Exception as e:
        logging.error(f"API request failed: {e}")
        return []

    try:
        json_start = output_text.find("[")
        json_end = output_text.rfind("]") + 1
        json_str = output_text[json_start:json_end]
        try:
            parsed = json.loads(json_str)
            logging.info(f"Parsed {len(parsed)} instructions from response.")
            return output_text
        except Exception as e:
            logging.error(f"Failed to parse JSON: {e}")
            return output_text
    except Exception as e:
        logging.error(f"Failed to parse JSON: {e}")
        return output_text

def cleanup_router_switch_prompts(text: str) -> str:
    """
    Removes router# or switch# prompts, including variations like:
    router(config)#, router(cfg)#, switch(config-if)#, etc.
    """
    try:
        pattern = r"(?:switch|router|Switch|Router)(?:\((?:config|cfg)[^)]*\))?#"
        return re.sub(pattern, "", text)
    except Exception as e:
        logging.error(f"Failed to cleanup router/switch prompts: {e}")
        logging.info(f"Original text: {text}")
        return ""

def chunk_and_process(text, out_file):
    chunks = chunk_text(text, max_tokens=1000)

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)}")
            instructions = generate_instructions(chunk, num_instructions=7)
            for instr in instructions:
                try:
                    cleaned_instr = cleanup_router_switch_prompts(instr)
                    fout.write(cleaned_instr)
                except Exception as e:
                    logging.error(f"Failed to write")

            logging.info(f"Finished chunk {i+1}/{len(chunks)}, wrote {len(instructions)} instructions.")

def process_text_files(input_dir: Path):
    """
    Iterates through all *.txt files in input_dir, cleans up router/switch prompts,
    and writes cleaned files to output_dir with the same filename.
    """
    for file_path in input_dir.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            cleaned_text = cleanup_router_switch_prompts(text)

            output_file = output_dir / file_path.name
            output_file.write_text(cleaned_text, encoding="utf-8")

            print(f"✅ Processed: {file_path.name} → {output_file}")
        except Exception as e:
            print(f"❌ Failed to process {file_path.name}: {e}")

def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Generate pretraining instructions from a text file using Ollama.")
    parser.add_argument("--input-dir", required=True, help="Path to input text file directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = "QA"
    os.makedirs(out_dir, exist_ok=True)

    for input_file in input_dir.glob("*.txt"):
        out_file = os.path.join(out_dir, os.path.basename(input_file) + ".jsonl")
        logging.info(f"Reading input file: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunk_and_process(text, out_file)
        logging.info(f"All chunks processed. Output saved to {out_file}")


if __name__ == "__main__":
    main()
