import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from app import SOURCES

def verify_yield():
    print("🔍 Verifying Scraper Yields (max_pages=3)...")
    total = 0
    
    # Run sequentially to debug individual failures
    for name, func in SOURCES.items():
        try:
            print(f"   > Scraping {name}...", end=" ", flush=True)
            headlines = func(max_pages=3)
            count = len(headlines)
            print(f"Found {count} headlines.")
            total += count
        except Exception as e:
            print(f"❌ Failed: {e}")

    print(f"\n✅ Total Raw Headlines: {total}")

if __name__ == "__main__":
    verify_yield()
