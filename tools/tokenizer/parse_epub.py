import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
import shutil
import re
from secure_do_not_commit.text_normalizer import normalize_line

# # Import global normalization function
# try:
#     from secure_do_not_commit.text_normalizer import normalize_line
# except ImportError:
#     # Fallback if import fails
#     def normalize_line(text: str, apply_content_formatting: bool = False) -> str:
#         return text

# MESSAGE_PATTERNS = [
#     (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),       # Pointers
#     (re.compile(r"\b\d+\.\d+\b"), "<FLOAT>"),       # Floats
#     (re.compile(r"\b\d+ULL\b"), "<INT>"),           # Unsigned long long
#     (re.compile(r"\b\d+L\b"), "<INT>"),            # Long ints
#     (re.compile(r"\b\d+\b"), "<INT>"),              # Integers
#     (re.compile(r"[A-Za-z0-9_\-]+mqipc[A-Za-z0-9_\-]*"), "<STR>"),  # mqipc-like strings
# ]


def normalize_log_line(line: str) -> str:
    """
    Replaces timestamps and hex patterns with tokens using global normalization.
    """
    #return normalize_line(line, apply_content_formatting=False)
    return line.strip()


def normalize_text(tag) -> str:
    """
    Extracts text preserving newlines in <pre>/<code> and logical structure for others.
    Removes decorative lines like '----- ------' or '==== ======'.
    """
    if tag.name in ["pre", "code"]:
        raw_text = tag.get_text()
        lines = raw_text.splitlines()

        # Match lines that are mostly decorative (repeated characters with optional spaces)
        def is_decorative(line: str) -> bool:
            # Remove spaces to normalize
            no_space = line.replace(" ", "")
            # Line must be at least 10 characters long
            if len(no_space) < 10:
                return False
            # Check if line is purely made of decoration chars (no letters/numbers)
            if re.fullmatch(r"[-=~_*]+", no_space) and not re.search(r"\w", line):
                return True
            return False


        clean_lines = [normalize_log_line(line) for line in lines if not is_decorative(line)]
        return "\n".join(clean_lines).strip()

    else:
        return tag.get_text(separator=" ").strip()

def split_by_size(content, size_kb, title, balance_tag=None, enclosure_tag="DOC"):
    # Split content into chunks of approximately size_kb kilobytes
    chunk_size = size_kb * 1024  # Convert KB to bytes
    chunks = []
    start = 0
    content_bytes = content.encode('utf-8')

    while start < len(content_bytes):
        target_end = start + chunk_size
        if target_end >= len(content_bytes):
            # Last chunk
            chunk_bytes = content_bytes[start:]
            end = len(content_bytes)
        else:
            # Try to find a good break point (newline) near the target size
            search_start = max(start, target_end - 512)  # Look back up to 512 bytes
            search_end = min(len(content_bytes), target_end + 512)  # Look ahead up to 512 bytes

            # Find the last newline in the search area
            newline_pos = content_bytes.rfind(b'\n', search_start, search_end)
            if newline_pos != -1 and newline_pos > start:
                end = newline_pos + 1
            else:
                end = target_end

            chunk_bytes = content_bytes[start:end]

        chunk = chunk_bytes.decode('utf-8', errors='ignore')
        if enclosure_tag:
            chunk = f"""
        <{enclosure_tag}>
        <TITLE>{title}</TITLE>
        {chunk}
        </{enclosure_tag}>
        """
        if chunk.strip():
            chunks.append(chunk)
        start = end

    return chunks



def parse_epub_file(epub_file: Path) -> str:
    """Extract structured content from a single EPUB file and return as <DOC> block."""
    book = epub.read_epub(str(epub_file))
    title = book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else epub_file.stem

    docs = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text_blocks = []
            skip_content = False
            
            # Get all relevant tags in order
            all_tags = soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"])
            
            i = 0
            while i < len(all_tags):
                tag = all_tags[i]
                text = normalize_text(tag)
                
                # Skip decorative lines and empty content
                if re.match(r"^[-=~_*]{10,}$", text) or not text.strip():
                    i += 1
                    continue

                if tag.name in ["h1"]:
                    # If the text is "Contents" then we need to skip all the subsequent text 
                    # Till we encounter another section.
                    if text == "Contents":
                        skip_content = True
                        i += 1
                        continue
                    elif "Cisco Trademarks" in text:
                        skip_content = True
                    else:
                        skip_content = False
                        
                elif tag.name in ["h2", "h3"] and not skip_content:
                    # Check for duplication with previous text block
                    # formatted_text = f"<SUBSECTION>{text}</SUBSECTION>"
                    formatted_text = f"{text}"
                    if not text_blocks or text_blocks[-1] != formatted_text:
                        text_blocks.append(formatted_text)
                elif tag.name == "p" and not skip_content:
                    # Additional check for paragraph content to avoid empty paragraphs and duplicates
                    if text.strip():
                        # Check for duplication with previous text block
                        if not text_blocks or text_blocks[-1] != text:
                            text_blocks.append(text)
                elif tag.name == "li" and not skip_content:
                    # Check for duplication with previous text block
                    formatted_text = f"- {text}"
                    if not text_blocks or text_blocks[-1] != formatted_text:
                        text_blocks.append(formatted_text)
                elif tag.name in ["pre", "code"] and not skip_content:
                    # Group consecutive code/pre tags together
                    code_blocks = [text]  # Start with current tag's text
                    j = i + 1
                    
                    # Look ahead for consecutive code/pre tags
                    while j < len(all_tags):
                        next_tag = all_tags[j]
                        if next_tag.name in ["pre", "code"]:
                            next_text = normalize_text(next_tag)
                            if next_text.strip():  # Only add non-empty code blocks
                                code_blocks.append(next_text)
                            j += 1
                        else:
                            # Stop when we hit a non-code tag
                            break
                    
                    # Deduplicate lines within the combined code blocks
                    all_lines = []
                    for code_block in code_blocks:
                        all_lines.extend(code_block.split('\n'))
                    
                    # Remove duplicate consecutive lines and empty lines
                    deduplicated_lines = []
                    prev_line = None
                    for line in all_lines:
                        clean_line = line.strip()
                        if clean_line and clean_line != prev_line:
                            deduplicated_lines.append(line)
                            prev_line = clean_line
                        elif not clean_line and prev_line is not None:
                            # Keep empty lines only if they're not at the start and previous wasn't empty
                            if deduplicated_lines and deduplicated_lines[-1].strip():
                                deduplicated_lines.append(line)
                    
                    # Combine deduplicated lines
                    if deduplicated_lines:
                        combined_code = "\n".join(deduplicated_lines)
                        text_blocks.append(f"<CODE>\n{combined_code}\n</CODE>")
                    
                    # Skip the tags we've already processed (j-1 because we'll increment i at end of loop)
                    i = j - 1

                i += 1

            if text_blocks:
                doc_content = f"<SECTION>\n" + "\n".join(text_blocks) + "\n</SECTION>"
                #doc_content = "\n\n".join(text_blocks) + "\n"
                docs.append(doc_content)

    all_doc_content = "\n".join(docs)
    return f"\n{all_doc_content}\n", title


def process_epub_directory(input_dir: Path, output_dir: Path):
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    merged_docs = []
    for epub_file in input_dir.rglob("*.epub"):
        try:
            doc_text, title = parse_epub_file(epub_file)
            if not doc_text.strip():
                print(f"⚠️ No content extracted from {epub_file.name}")
                continue

            # Parse each line in the doc_text to make sure there is no empty line
            doc_text = "\n".join(line.strip() for line in doc_text.splitlines() if line.strip())
            if not doc_text:
                print(f"⚠️ No valid content in {epub_file.name}")
                continue

            file_chunks = split_by_size(doc_text, 50, title, 
                          balance_tag="SECTION", enclosure_tag="DOC")  # Example: split into 500KB chunks if needed
            
            for idx, chunk in enumerate(file_chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Splitting mode, use numbered files
                out_file = output_dir / f"{epub_file.stem}_split_{idx+1}.txt"
                with open(out_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(chunk)

            #out_file = staging_dir / (epub_file.stem + ".txt")
            #out_file.write_text(doc_text, encoding="utf-8")
            merged_docs.append(doc_text)

            print(f"✅ Processed {epub_file.name} → {out_file.name} split into {len(file_chunks)} chunks")
        except Exception as e:
            print(f"❌ Failed to process {epub_file.name}: {e}")

    # Merge into final file
    # merged_file = output_dir / "merged_docs.txt"
    # with open(merged_file, "w", encoding="utf-8") as f:
    #     for doc in merged_docs:
    #         f.write(doc + "\n\n")

    #print(f"\n🎉 Merged {len(merged_docs)} EPUB files → {merged_file}")

    # Optional cleanup
    # shutil.rmtree(staging_dir)
    print(f"🗑️ Deleted staging directory: {staging_dir}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Normalize EPUB files into structured <DOC> format.")
#     parser.add_argument("input_dir", type=str, help="Directory containing EPUB files")
#     parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

#     args = parser.parse_args()
#     process_epub_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
#     # Merge into final file
#     merged_file = args.output_dir / "merged_docs.txt"
#     with open(merged_file, "w", encoding="utf-8") as f:
#         for doc in merged_docs:
#             f.write(doc + "\n\n")

#     print(f"\n🎉 Merged {len(merged_docs)} EPUB files → {merged_file}")

#     # Optional cleanup
#     # shutil.rmtree(staging_dir)
#     print(f"🗑️ Deleted staging directory: {staging_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize EPUB files into structured <DOC> format.")
    parser.add_argument("input_dir", type=str, help="Directory containing EPUB files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_epub_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
