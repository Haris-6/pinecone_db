import pdfplumber #his library is used to read PDF files and extract text from them
from config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or "" #it extracts text from each page of the PDF and handles cases where a page might be empty.
            pages.append(text)
    return "\n\n".join(pages) #Combines all pages’ text into a single string, separated by two newlines.

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + size
        chunk = text[start:end]
        chunks.append((chunk, start, min(end, length))) #What it does: Adds a tuple to the list: (chunk_text, start_index, end_index).
        start += size - overlap #Moves the start pointer forward for the next chunk. and Keeps a portion of the previous chunk (overlap) in the next chunk to maintain context.

    return chunks
