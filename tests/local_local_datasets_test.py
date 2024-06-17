from datasets import load_dataset

dataset_path = ""


def extract_field(example):
    if "text" in example:
        res = example["text"]
    else:
        res = example["content"]
    return {"text": res}


dataset = load_dataset(dataset_path, split="train", streaming=True)
dataset = dataset.map(extract_field)

for data in dataset:
    print(data["text"])
