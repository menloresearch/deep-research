import pandas as pd
from datasets import load_dataset

ds = load_dataset("callanwu/WebWalkerQA", split="main")
df = ds.to_pandas()
# change column names: question -> query, answer -> gold_answer
df.rename(columns={"question": "query", "answer": "gold_answer"}, inplace=True)
df.to_csv("webwalkerqa_main.csv", index=False)
