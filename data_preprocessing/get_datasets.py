import requests
from tqdm import tqdm
from pathlib import Path
import shutil

def download_file(url, folder, project_root):
    if not folder.exists():
        folder.mkdir(parents=True)
    
    filename = url.split('/')[-1]
    file_path = folder / filename
    
    print(f"Connecting to {url}...")
    # Get remote file size for comparison/display
    try:
        # Define headers to look like a browser, helps avoid some 503/403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Use a timeout to prevent hanging indefinitely
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
    except Exception as e:
        print(f"Error fetching metadata for {filename}: {e}")
        return

    if file_path.exists():
        local_size = file_path.stat().st_size
        if total_size > 0 and local_size == total_size:
            print(f"File {filename} ({local_size/1024/1024:.2f} MB) already exists and is complete. Skipping...")
            return
        elif total_size == 0:
            print(f"File {filename} exists. Cannot verify size because server didn't provide content-length. Skipping to be safe...")
            return
        else:
            print(f"File {filename} exists but size mismatch ({local_size} vs {total_size} bytes). Redownloading...")

    print(f"Downloading {filename}...")
    try:
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ascii=False,
            ncols=100
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                if data:
                    size = f.write(data)
                    bar.update(size)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        # Only delete if it's a new file or we explicitly wanted to overwrite
        # To be safe, we might not want to delete it if it was already partially there
        pass 

def main():
    # Use paths relative to the script location
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    links_file = project_root / 'datasetLink.txt'
    target_folder = project_root / 'dataset'

    if not links_file.exists():
        print(f"Error: {links_file} not found.")
        return

    with open(links_file, 'r') as f:
        links = [line.strip() for line in f if line.strip()]

    print(f"Starting dataset download script. Found {len(links)} links.")
    for link in links:
        download_file(link, target_folder, project_root)

if __name__ == "__main__":
    main()
