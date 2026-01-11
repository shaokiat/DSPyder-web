import requests
import os
from pathlib import Path
import time

# List of identified HDB URLs
HDB_URLS = [
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/enhanced-cpf-housing-grant-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/enhanced-cpf-housing-grant-singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/financing-a-flat",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-new-flats/application/priority-schemes"
]

def download_hdb_pages(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Starting HDB documentation download to {output_dir}...")
    
    for url in HDB_URLS:
        try:
            name = url.split("/")[-1] + ".html"
            target_path = output_dir / name
            
            print(f"Fetching: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            target_path.write_text(response.text, encoding="utf-8")
            print(f"  Saved to: {target_path}")
            
            # Be polite to the server
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error downloading {url}: {e}")

if __name__ == "__main__":
    # Save to a dedicated directory for HDB data
    BASE_DIR = Path(__file__).parent.parent
    HDB_RAW_DIR = BASE_DIR / "data" / "hdb_raw"
    
    download_hdb_pages(HDB_RAW_DIR)
