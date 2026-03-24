import os
import requests
import pdfminer
from io import BytesIO
from pdfminer.high_level import extract_text

BASE_DIR = "premchand_corpus"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
TXT_DIR = os.path.join(BASE_DIR, "texts")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

# ---------------------------
# Step 1: Download PDFs
# ---------------------------
def download_pdf(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    
    pdf_path = os.path.join(PDF_DIR, filename)
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    
    return pdf_path

# ---------------------------
# Step 2: Extract text
# ---------------------------
def extract_pdf_text(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return ""

# ---------------------------
# Step 3: Clean text
# ---------------------------
def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    
    for line in lines:
        line = line.strip()
        
        # Remove noise
        if not line:
            continue
        if len(line) < 20:
            continue
        if any(x in line.lower() for x in ["page", "www", "http", "copyright"]):
            continue
        
        cleaned.append(line)
    
    return "\n".join(cleaned)

# ---------------------------
# Step 4: Save text
# ---------------------------
def save_text(filename, text):
    path = os.path.join(TXT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------
# Step 5: Sources (PDF links)
# ---------------------------
sources = {
    "karmabhumi.pdf": "https://www.hindustanbooks.com/books/karmbhumi/karmbhumi.pdf",
    "gaban.pdf": "https://www.hindustanbooks.com/books/gaban/gaban.pdf",
}

# ---------------------------
# Pipeline execution
# ---------------------------
for pdf_name, url in sources.items():
    try:
        print(f"Downloading {pdf_name}...")
        pdf_path = download_pdf(url, pdf_name)
        
        print(f"Extracting {pdf_name}...")
        raw_text = extract_pdf_text(pdf_path)
        
        print(f"Cleaning {pdf_name}...")
        cleaned = clean_text(raw_text)
        
        txt_name = pdf_name.replace(".pdf", ".txt")
        save_text(txt_name, cleaned)
        
        print(f"Saved {txt_name}\n")
        
    except Exception as e:
        print(f"Error processing {pdf_name}: {e}")

# ---------------------------
# Step 6: Combine corpus
# ---------------------------
combined_path = os.path.join(BASE_DIR, "combined_corpus.txt")

with open(combined_path, "w", encoding="utf-8") as outfile:
    for file in os.listdir(TXT_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(TXT_DIR, file), encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n\n")

print("✅ Corpus ready at:", combined_path)