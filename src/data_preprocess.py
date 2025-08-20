import re
from collections import Counter

def clean_text(raw_text):
    # Split text into lines
    lines = raw_text.splitlines()

    # Count line frequency to identify repeated headers/footers
    line_counts = Counter(lines)

    cleaned_lines = []
    seen_lines = set()
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        # Skip if line appears too often (likely header/footer)
        if line_counts[line] > 3:  # adjust threshold
            continue
        # Skip page numbers
        if re.match(r'^\s*(page\s*\d+|\d+\s*of\s*\d+|\d+)\s*$', line, re.IGNORECASE):
            continue
        # Skip lines that look like headers with patterns like "--- filename | Page number ---"
        if re.match(r'^---.*Page\s*\d+.*---$', line, re.IGNORECASE):
            continue
        # Skip duplicate lines
        line_stripped = line.strip()
        if line_stripped in seen_lines:
            continue
        seen_lines.add(line_stripped)
        cleaned_lines.append(line_stripped)

    # Join lines and normalize spaces
    text = " ".join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_sections(text):
    # Define section headings to look for
    sections = ["Income Statement", "Balance Sheet", "Cash Flow Statement", "Notes"]
    # Create a regex pattern to split on these headings
    pattern = re.compile(r'(' + '|'.join([re.escape(section) for section in sections]) + r')', re.IGNORECASE)
    # Split text by the headings, keeping the headings
    parts = pattern.split(text)
    segmented = {}
    current_section = None
    for part in parts:
        part_strip = part.strip()
        # Check if this part is a section heading
        if any(part_strip.lower() == sec.lower() for sec in sections):
            current_section = part_strip
            segmented[current_section] = ""
        elif current_section:
            segmented[current_section] += part_strip + " "
    # Strip trailing spaces from each section text
    for key in segmented:
        segmented[key] = segmented[key].strip()
    return segmented

def split_into_chunks(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunk = {
            "id": f"chunk_{i//chunk_size}",
            "text": chunk_text,
            "metadata": {
                "chunk_index": i // chunk_size,
                "chunk_size": len(chunk_words)
            }
        }
        chunks.append(chunk)
    return chunks

with open("data/preprocessed/combined_text.txt", "r") as f:
    data_preprocess_code = f.read()

prepreocessed_data = clean_text(data_preprocess_code)

segmented_data = segment_sections(prepreocessed_data)
    
#print("Data Preprocessed file:\n", prepreocessed_data)

for section, content in segmented_data.items():
    print(f"Section: {section}, Length: {len(content)}")

chunks_100 = split_into_chunks(prepreocessed_data, 100)
chunks_400 = split_into_chunks(prepreocessed_data, 400)

print(f"Number of 100-word chunks: {len(chunks_100)}")
if chunks_100:
    print("First 100-word chunk example:", chunks_100[0])

print(f"Number of 400-word chunks: {len(chunks_400)}")
if chunks_400:
    print("First 400-word chunk example:", chunks_400[0])
