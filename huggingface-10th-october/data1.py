from datasets import load_dataset

ds = load_dataset("imdb")
print(ds)                      # shows train/test splits
print(ds["train"][0])          # first example

small = ds["train"].shuffle(seed=42).select(range(1000))
print(len(small))
