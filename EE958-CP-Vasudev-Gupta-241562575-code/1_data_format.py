#!/usr/bin/env python
# coding: utf-8

import json
import os
import warnings
warnings.filterwarnings('ignore')


# Data Paths
raw_data_path       = os.path.join("data", "raw")
processed_data_path = os.path.join("data", "processed")
os.makedirs(processed_data_path, exist_ok=True)

# Load the raw data
train_data = json.load(open(os.path.join("data", "raw", "train_data1.json")))
val_data = json.load(open(os.path.join("data",   "raw",   "val_data1.json")))


# JSONL Files

# English to Hindi
with open(os.path.join(processed_data_path, "eng2hindi_train.jsonl"), "w", encoding="utf-8") as f:
    for record in [v for _, v in train_data['English-Hindi']['Train'].items()]:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open(os.path.join(processed_data_path, "eng2hindi_val.jsonl"), "w", encoding="utf-8") as f:
    for record in [v for _, v in val_data['English-Hindi']['Validation'].items()]:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# English to Bengali
with open(os.path.join(processed_data_path, "eng2bengali_train.jsonl"), "w", encoding="utf-8") as f:
    for record in [v for _, v in train_data['English-Bengali']['Train'].items()]:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open(os.path.join(processed_data_path, "eng2bengali_val.jsonl"), "w", encoding="utf-8") as f:
    for record in [v for _, v in val_data['English-Bengali']['Validation'].items()]:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
