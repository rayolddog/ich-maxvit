"""
Download the trained MaxViT checkpoint from HuggingFace Hub.
Run once from the ich-maxvit folder:
    python download_checkpoint.py
"""
from huggingface_hub import hf_hub_download
from pathlib import Path

dest = Path(__file__).parent / "checkpoints_maxvit"
dest.mkdir(exist_ok=True)

print("Downloading best_maxvit_ich.pth (1.4 GB) from HuggingFace Hub...")
hf_hub_download(
    repo_id   = "brodown3/ich-maxvit",
    filename  = "best_maxvit_ich.pth",
    local_dir = str(dest),
)
print(f"Saved to {dest / 'best_maxvit_ich.pth'}")
print("Ready to run: python run_demo_direct.py")
