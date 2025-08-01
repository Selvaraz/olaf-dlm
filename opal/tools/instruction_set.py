import os
import json
import argparse
from openai import OpenAI

# ✅ Prompt Template
BASE_PROMPT = """
You are an expert Cisco network engineer and technical writer. You specialize in creating high-quality instruction-tuning datasets for training domain-specific LLMs.

Your goal is to read the following Cisco documentation text and produce instruction JSON objects for fine-tuning a domain-specific LLM.

Each JSON object must follow one of these actions:
- orbit.teach → Step-by-step guide explaining how to configure a feature.
- orbit.configure → Commands to configure a feature.
- orbit.show → Commands to view status or verify configuration.
- orbit.explain → Detailed explanation of a feature or concept.
- orbit.troubleshoot → Troubleshooting steps for an issue.
- orbit.summary → Concise summary of the topic.

Important Rules:
- Do NOT include CLI commands in the prompt. The prompt must describe the functionality in natural language.
- The response for orbit.configure and orbit.show must contain correct CLI commands in an array under "commands".
- For orbit.teach and orbit.troubleshoot, the "response" must contain an array of steps with brief descriptions.
- For orbit.explain, the "response" must contain "explanation" with a detailed step by step description.
- For orbit.summary, the "response" must contain "summary" with a concise summary for the given set of logs (or) command output on the Cisco devices.
- Commands starting with debug → Use orbit.enable_debug as the "action".
- The dataset must be in valid JSONL format, with one JSON object per line.
- Include at least 5 samples per feature/topic (teach, configure, show, explain, troubleshoot, summary).

Documentation Text:
"""

def chunk_text(text, max_chars=4000):
    """Split large text into manageable chunks for the API."""
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_instructions(client, input_text):
    """Send prompt to OpenAI API and return JSON objects."""
    prompt = BASE_PROMPT + "\n" + input_text
    response = client.chat.completions.create(
        model="gpt-4o",  # You can also use "gpt-4o-mini" to save cost
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate instruction-tuning dataset from Cisco docs.")
    parser.add_argument("--input-doc", required=True, help="Path to the input documentation file (text format)")
    parser.add_argument("--output-file", default="instruction_dataset.jsonl", help="Path to save JSONL output")
    args = parser.parse_args()

    # ✅ Set your OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set the OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    # ✅ Read input file
    with open(args.input_doc, "r") as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)
    print(f"🔹 Total Chunks to Process: {len(chunks)}")

    with open(args.output_file, "w") as out:
        for i, chunk in enumerate(chunks, 1):
            print(f"➡️ Processing chunk {i}/{len(chunks)} ...")
            try:
                result_text = generate_instructions(client, chunk)

                # ✅ Extract valid JSON objects (handle JSONL)
                for line in result_text.split("\n"):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            json.loads(line)  # Validate JSON
                            out.write(line + "\n")
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping invalid JSON line in chunk {i}")
            except Exception as e:
                print(f"❌ Error in chunk {i}: {e}")

    print(f"✅ Finished! JSONL saved to {args.output_file}")

if __name__ == "__main__":
    main()
