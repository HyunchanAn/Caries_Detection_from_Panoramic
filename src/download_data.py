from huggingface_hub import snapshot_download
import os

def download_dentex_dataset(local_dir):
    """
    Downloads the DENTEX dataset from Hugging Face Hub.
    Repo ID: ibrahimhamamci/DENTEX
    """
    repo_id = "ibrahimhamamci/DENTEX"
    
    print(f"Downloading {repo_id} to {local_dir}...")
    
    try:
        # Download the entire repository
        # This dataset usually contains .zip files or directories with images
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Download complete!")
        print(f"Files are saved in: {local_dir}")
        print("Please unzip any .zip files if necessary.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DENTEX dataset")
    parser.add_argument("--dir", type=str, default=None, help="Target directory for download")
    args = parser.parse_args()

    # Define download path
    if args.dir:
        DOWNLOAD_DIR = args.dir
    else:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
        DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DENTEX")
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    download_dentex_dataset(DOWNLOAD_DIR)
