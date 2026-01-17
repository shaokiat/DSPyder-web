import requests
import os
from pathlib import Path
import time
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse

# Seed URLs to start crawling from
SEED_URLS = [
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/financing-a-flat",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-new-flats/application/priority-schemes"
]

MAX_DEPTH = 3
MAX_PAGES = 50

def is_valid_url(url):
    """Check if URL is valid for crawling."""
    parsed = urlparse(url)
    # Only crawl HDB residential pages
    return (
        parsed.netloc == "www.hdb.gov.sg" and 
        parsed.path.startswith("/residential") and
        not any(parsed.path.lower().endswith(ext) for ext in ['.pdf', '.zip', '.jpg', '.png', '.mp4'])
    )

def download_hdb_pages(output_dir: Path, dry_run: bool = False):
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Queue stores (url, depth)
    queue = deque([(url, 0) for url in SEED_URLS])
    visited = set(SEED_URLS)
    pages_processed = 0
    
    mode_str = "DRY RUN" if dry_run else "LIVE"
    print(f"Starting HDB recursive crawl ({mode_str}) to {output_dir}...")
    print(f"Max depth: {MAX_DEPTH}, Max pages: {MAX_PAGES}")
    
    while queue and pages_processed < MAX_PAGES:
        url, depth = queue.popleft()
        
        try:
            indent = "  " * depth
            prefix = f"[{pages_processed+1}/{MAX_PAGES}]"
            print(f"{prefix} {indent}-> Fetching: {url}")
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            if not dry_run:
                # Save content
                # ... naming logic ...
                # improved naming to avoid collisions
                safe_name = urlparse(url).path.strip("/").replace("/", "_") + ".html"
                if not safe_name or safe_name == ".html":
                     safe_name = "index.html"
                
                target_path = output_dir / safe_name
                target_path.write_text(response.text, encoding="utf-8")
            
            pages_processed += 1
            
            # If we haven't reached max depth, find more links
            if depth < MAX_DEPTH:
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    next_url = urljoin(url, link['href'])
                    # Remove fragment identifier
                    next_url = next_url.split('#')[0]
                    
                    if next_url not in visited and is_valid_url(next_url):
                        visited.add(next_url)
                        queue.append((next_url, depth + 1))
            
            # Be polite
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error processing {url}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursively download HDB documentation.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without saving files")
    parser.add_argument("--output", type=str, help="Output directory path")
    
    args = parser.parse_args()
    
    # Save to a dedicated directory for HDB data
    project_root = Path(__file__).resolve().parent.parent.parent
    
    if args.output:
        HDB_RAW_DIR = Path(args.output)
    else:
        HDB_RAW_DIR = project_root / "data" / "hdb_raw"
    
    download_hdb_pages(HDB_RAW_DIR, dry_run=args.dry_run)
