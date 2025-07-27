import re
from pathlib import Path


# Format specifier to token mapping
FORMAT_MAP = {
    "%s": "<STR>",
    "%d": "<INT>",
    "%u": "<UINT>",
    "%x": "<HEX>",
    "%X": "<HEX>",
    "%f": "<FLOAT>",
    "%ld": "<LONG>",
    "%llu": "<ULL>",
    "%p": "<PTR>",
    "%c": "<CHAR>",
}


def replace_format_specifiers(fmt: str) -> str:
    """Replace C-style format specifiers with token placeholders."""
    fmt = fmt.replace("\\n", "").replace("\\t", " ")
    for spec, token in FORMAT_MAP.items():
        fmt = fmt.replace(spec, token)
    return fmt.strip()


def extract_module_name(text: str) -> str:
    """Extract the module name without 'btrace_' prefix."""
    match = re.search(r"module_def\s+btrace_([a-zA-Z0-9_]+)\s*{", text)
    return match.group(1) if match else "unknown_module"


def extract_traces(text: str):
    """Extract all traces from a single .trtdl file."""
    module_name = extract_module_name(text)

    pattern = re.compile(
        r'trace_def\s+__BT_[a-f0-9]+[\s\S]*?format_string\s+"(.*?)";[\s\S]*?function_name\s+"(.*?)";',
        re.MULTILINE,
    )
    matches = pattern.findall(text)

    return [
        f"{module_name}::{fn_name}: {replace_format_specifiers(fmt_str)}"
        for fmt_str, fn_name in matches
    ]


def process_all_trtdl_files_to_txt(input_dir: Path):
    """Process all .trtdl files and write individual .txt files to /txt_converted."""
    output_dir = input_dir / "txt_converted"
    output_dir.mkdir(exist_ok=True)

    trtdl_files = list(input_dir.rglob("*.trtdl"))
    if not trtdl_files:
        print("❗ No .trtdl files found.")
        return

    total_traces = 0

    for trtdl_file in trtdl_files:
        try:
            text = trtdl_file.read_text(encoding="utf-8")
            traces = extract_traces(text)
            total_traces += len(traces)

            output_txt = output_dir / (trtdl_file.stem + ".txt")
            with open(output_txt, "w", encoding="utf-8") as f:
                for line in traces:
                    f.write(line + "\n")

            print(f"✅ Wrote {len(traces)} traces to {output_txt.name}")
        except Exception as e:
            print(f"❌ Failed to process {trtdl_file.name}: {e}")

    print(f"\n🎉 Total {total_traces} trace lines extracted to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and tokenize .trtdl logs to txt files per file.")
    parser.add_argument("dir_path", type=str, help="Root directory containing .trtdl files.")

    args = parser.parse_args()
    process_all_trtdl_files_to_txt(Path(args.dir_path).resolve())

    print("\n✅ Done.")