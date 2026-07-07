from datasets import load_dataset
import re

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

"""
Things to clean:
- Make everything lowercase
- Handle empty strings and lines
- Handle <unk> endings
- Section headers
- Too many whitespaces or whitespaces before periods
- Weird characters, also not english
- @-@ needs to be -
"""

def clean(example):
    text = example["text"]

    # handles empty lines
    if not text.strip():
        return {"text": ""}
    # handles non ascii characters
    text = re.sub(r'[^\x00-\x7F]', '', text)
    # makes it all lowercase
    text = text.lower()
    # fixes one formatting issue
    text = text.replace("@-@", "-")
    # fixes leading and trailing whitespace
    text = text.strip()
    # remove headers
    text = re.sub(r'= .* =', '', text)        
    # fix spaced punctuation
    text = re.sub(r'\s([.,;:!?])', r'\1', text)  
    # fix spaced quotes
    text = re.sub(r'"\s(.*?)\s"', r'"\1"', text)  
    # fix spaced hyphens
    text = re.sub(r'\s-\s', '-', text)        
    # fix spaced possessives
    text = re.sub(r"\s's", "'s", text)   
    # fix spaced parentheses     
    text = re.sub(r'\(\s(.*?)\s\)', r'(\1)', text)  

    return {"text": text}

ds = ds.map(clean)
ds = ds.filter(lambda x: x["text"] != "")

ds = ds.train_test_split(test_size=0.1)

ds["train"].save_to_disk("cleaned_dataset_train")
ds["test"].save_to_disk("cleaned_dataset_test")