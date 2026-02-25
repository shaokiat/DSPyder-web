from bs4 import BeautifulSoup
import re
from pathlib import Path
import json

def load_html(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore")

def parse_html(html: str):
    return BeautifulSoup(html, "html.parser")

def strip_junk(soup):
    for tag in soup([
        "script", "style", "noscript",
        "header", "footer", "nav",
        "aside", "form"
    ]):
        tag.decompose()

def chunk_text_by_structure(soup, chunk_size=800, overlap=150):
    """
    Chunks text by respecting HTML structure (sections/paragraphs).
    Injects the section header into each chunk for better context.
    """
    chunks = []
    content = extract_main_content(soup)
    
    current_section = "General"
    current_chunk_text = ""
    
    # Process elements sequentially to maintain order and context
    for tag in content.find_all(["h1", "h2", "h3", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
            
        if tag.name.startswith("h"):
            # New section starts
            current_section = text
            # We don't necessarily start a new chunk here if the current one is small,
            # but usually headers are good break points.
            if current_chunk_text:
                # Flush current chunk
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "section": current_section
                })
                current_chunk_text = ""
        else:
            # Append text with section context prepended if chunk is new
            if not current_chunk_text:
                current_chunk_text = f"[{current_section}] "
            
            current_chunk_text += text + " "
            
            # If chunk exceeds size, save it and start new one with overlap
            if len(current_chunk_text) > chunk_size:
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "section": current_section
                })
                # Simple overlap: keep last N chars (ideally last sentence, but keeping it simple for now)
                overlap_text = current_chunk_text[-overlap:] if len(current_chunk_text) > overlap else ""
                current_chunk_text = f"[{current_section}] ... {overlap_text}"

    # Final flush
    if current_chunk_text:
        chunks.append({
            "text": current_chunk_text.strip(),
            "section": current_section
        })
        
    return chunks

def extract_main_content(soup):
    for tag in ["main", "article"]:
        content = soup.find(tag)
        if content:
            return content
    return soup.body or soup


def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def html_to_clean_text(path):
    html = load_html(path)
    soup = parse_html(html)
    strip_junk(soup)

    content = extract_main_content(soup)
    text = content.get_text(separator=" ")

    return normalize_text(text)

def extract_with_headings(soup):
    content = extract_main_content(soup)

    lines = []
    for tag in content.find_all(["h1", "h2", "h3", "p", "li"]):
        if tag.name.startswith("h"):
            header_text = tag.get_text(strip=True)
            if header_text:
                lines.append(f"___SECTION_START___ {header_text} ___SECTION_END___")
        else:
            text = tag.get_text(" ", strip=True)
            if text:
                lines.append(text)

    return " ".join(lines)

def process_directory(source_dir: Path, chunks_output_path: Path, source_name: str = "Knowledge Base"):
    chunks_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_files = list(source_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {source_dir}")
    
    # Calculate base project path for relative paths
    project_root = Path(__file__).parent.parent.parent.parent
    
    all_chunks = []
    
    for html_path in html_files:
        try:
            # Calculate source_path relative to project root
            try:
                relative_source_path = str(html_path.relative_to(project_root))
            except ValueError:
                relative_source_path = str(html_path)

            html_content = load_html(html_path)
            soup = parse_html(html_content)
            strip_junk(soup)
            
            clean_text = extract_with_headings(soup)
            
            doc_id = html_path.stem.replace("-", "_")
            
            # Perform improved chunking
            chunks_meta = chunk_text_by_structure(soup)
            for i, chunk in enumerate(chunks_meta):
                all_chunks.append({
                    "chunk_id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "source": source_name,
                    "source_path": relative_source_path,
                    "section": chunk["section"],
                    "text": chunk["text"]
                })
                
            print(f"Processed: {html_path.name} -> Added {len(chunks_meta)} chunks")
        except Exception as e:
            print(f"Error processing {html_path.name}: {e}")

    # Save consolidated chunks
    chunks_output_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_chunks)} chunks to {chunks_output_path}")

if __name__ == "__main__":
    import os
    # Default to current project structure
    CURRENT_DIR = Path(__file__).parent
    # Data is in root data/ (2 levels up from src/ingestion/)
    PROJECT_ROOT = CURRENT_DIR.parent.parent
    BASE_DATA_DIR = PROJECT_ROOT / "data"
    
    # Configuration via Environment Variables
    SOURCE_DIR = Path(os.getenv("SOURCE_DIR", BASE_DATA_DIR / "hdb_raw"))
    CHUNKS_OUTPUT = Path(os.getenv("CHUNKS_OUTPUT", BASE_DATA_DIR / "chunks.json"))
    SOURCE_NAME = os.getenv("SOURCE_NAME", "HDB Housing Guide")
    
    print(f"--- Data Ingestion Pipeline ---")
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Output File:      {CHUNKS_OUTPUT}")
    print(f"Source Name:      {SOURCE_NAME}")
    print(f"-------------------------------")
    
    process_directory(SOURCE_DIR, CHUNKS_OUTPUT, SOURCE_NAME)
