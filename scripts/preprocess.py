from datasets import load_dataset

ds = load_dataset("kensho/DocFinQA", split="train[:3]")  # first 10 examples only
print(ds[0])