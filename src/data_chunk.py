# Simple sentence splitter using regex
# ------------------------------
# Step 1: Simple sentence tokenizer
# ------------------------------
import re
import uuid

def simple_sent_tokenize(text):
    # Split on periods, question marks, exclamation marks, keeping punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty strings
    return [s.strip() for s in sentences if s.strip()]


# ------------------------------
# Step 2: Chunking function
# ------------------------------
def create_chunks(sentences, chunk_sizes=[100, 400], source_file="unknown"):
    """
    Create overlapping chunks of sentences based on token limits.
    Assigns unique ID and metadata to each chunk.
    """
    all_chunks = []

    for chunk_size in chunk_sizes:
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())  # token count approximation
            if current_length + sentence_length > chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = str(uuid.uuid4())
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "source_file": source_file,
                    "chunk_size": chunk_size
                })
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Save last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = str(uuid.uuid4())
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source_file": source_file,
                "chunk_size": chunk_size
            })

    return all_chunks


# ------------------------------
# Step 3: Load text files
# ------------------------------
inputfile = "data/preprocessed/combined_text.txt"

with open(inputfile, "r", encoding="utf-8") as f:
    raw_file = f.read()

# ------------------------------
# Step 4: Tokenize into sentences
# ------------------------------
processed_file = simple_sent_tokenize(raw_file)

# ------------------------------
# Step 5: Create chunks
# ------------------------------

chunk_100 = create_chunks(processed_file, chunk_sizes=[100], source_file=inputfile) # Use inputfile here
chunk_400 = create_chunks(processed_file, chunk_sizes=[400], source_file=inputfile) # Use inputfile here

# ------------------------------
# Step 6: Quick check
# ------------------------------
print(f"2023-24 and 2024-25 report contains sentences: {len(processed_file)} \n")
print(f"100 size chunk contains: {len(chunk_100)} sentences\n")
print(f"400 size chunk contains: {len(chunk_400)} sentences\n")
