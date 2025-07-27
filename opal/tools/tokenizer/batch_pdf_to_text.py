import os
from pathlib import Path
from pdfminer.high_level import extract_text
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed


def clean_text(text: str) -> str:
    """Clean up extracted text by normalizing whitespace."""
    lines = text.splitlines()
    cleaned = [" ".join(line.strip().split()) for line in lines if line.strip()]
    return "\n".join(cleaned)


def convert_single_pdf(pdf_path: Path, output_dir: Path):
    """Convert a single PDF to a text file."""
    try:
        print(f"Processing: {pdf_path.name}")
        text = extract_text(str(pdf_path))
        cleaned = clean_text(text)

        output_file = output_dir / (pdf_path.stem + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to process {pdf_path.name}: {e}")


def convert_all_pdfs_in_dir(input_dir: str, max_workers: int = None):
    """Iterate through all PDFs in the given directory and convert to text using processes."""
    input_path = Path(input_dir).resolve()

    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"{input_path} is not a valid directory.")

    output_dir = input_path / "txt_converted"
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting conversion using processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_single_pdf, pdf_file, output_dir): pdf_file
            for pdf_file in pdf_files
        }

        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Exception while processing {pdf_file.name}: {e}")

    print(f"\n✅ Completed processing all PDFs.")


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Multithreaded PDF to text converter")
    parser.add_argument("dir_path", type=str, help="Directory containing PDF files.")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(),
                        help="Number of threads to use (default: number of CPU cores)")

    args = parser.parse_args()
    print("\nArguments:")
    print("----------")
    print("{:<20} {:<20}".format("Argument", "Value"))
    print("----------")
    for arg, value in vars(args).items():
        print("{:<20} {:<20}".format(arg, value))
    print("")
    convert_all_pdfs_in_dir(args.dir_path, args.threads)
