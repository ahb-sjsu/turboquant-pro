#!/usr/bin/env python3
"""Find loadable aesthetic datasets on HuggingFace."""
from datasets import load_dataset

datasets_to_try = [
    ("tonyassi/ava-aesthetic-nima-scores", "train"),
    ("Shuai1995/TAD66K_for_Image_Aesthetics_Assessment", "train"),
    ("logasja/mit-adobe-fivek", "train"),
]

for name, split in datasets_to_try:
    print(f"Trying {name}...")
    try:
        ds = load_dataset(name, split=split, streaming=True)
        sample = next(iter(ds))
        cols = list(sample.keys())
        types = [(k, type(v).__name__) for k, v in sample.items()]
        print(f"  OK! Columns: {cols}")
        print(f"  Types: {types}")
        # Check if there's a score/rating field
        for k, v in sample.items():
            if isinstance(v, (int, float)):
                print(f"  Numeric: {k} = {v}")
    except Exception as e:
        print(f"  FAILED: {e}")
    print()

# Also: can we just use the ethics corpus embeddings we already have?
# They have no beauty ratings, but we can test eigenspectrum structure
print("Ethics corpus already available in PostgreSQL (2.4M embeddings)")
print("Can test: eigenspectrum variation across traditions")
print("Can test: D_eff distribution within vs across traditions")

# Alternative: generate our own CLIP embeddings for AVA images
# But that requires downloading 255K images (~32GB) which is slow
# Better: use a dataset that already has embeddings
print()
print("Best option: download LAION aesthetics subset with CLIP embeddings")
print("  huggingface.co/datasets/laion/laion2B-en-aesthetic")
print("  Has pre-computed CLIP embeddings + predicted aesthetic scores")
