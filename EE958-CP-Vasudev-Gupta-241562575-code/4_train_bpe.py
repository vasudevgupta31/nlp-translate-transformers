#!/usr/bin/env python
# coding: utf-8

import json
import os
import multiprocessing as mp
import time
from src.tokenizers.bpe.eng import MyBPETokenizer
from src.tokenizers.bpe.indic import IndicBPETokenizer

os.makedirs("tokenizers", exist_ok=True)

def load_translation_data(filepath):
    """loads up the translation pairs from jsonl"""
    en_texts = []
    target_texts = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            en_texts.append(data['source'])
            target_texts.append(data['target'])
    
    return en_texts, target_texts

def train_english_tokenizer(all_english_texts, vocab_size, max_length):
    """train english bpe tokenizer - runs in its own process"""
    print(f"[PID {os.getpid()}] starting english tokenizer training.")
    print("="*60)
    
    # create and train the tokenizer
    eng_tokenizer = MyBPETokenizer(vocab_size=vocab_size, max_len=max_length)
    eng_vocab_size = eng_tokenizer.train_bpe(all_english_texts)

    # save it 
    eng_tokenizer_path = os.path.join("tokenizers", f"english_shared_bpe_{vocab_size}.pkl")
    eng_tokenizer.save_tokenizer(eng_tokenizer_path)
    
    print(f"[English] done! got vocab size: {eng_vocab_size}")
    return eng_vocab_size

def train_hindi_tokenizer(hi_texts, vocab_size, max_length):
    """train hindi bpe tokenizer - seperate process"""
    print(f"[PID {os.getpid()}] starting hindi tokenizer.")
    print("="*60)
    
    hi_tokenizer = IndicBPETokenizer(vocab_size=vocab_size, max_len=max_length)
    hi_vocab_size = hi_tokenizer.train_bpe(hi_texts)
    
    # save the trained tokenizer
    hi_tokenizer_path = os.path.join("tokenizers", f"hindi_bpe_{vocab_size}.pkl")
    hi_tokenizer.save_tokenizer(hi_tokenizer_path)
    
    print(f"[Hindi] finished training! vocab size: {hi_vocab_size}")
    return hi_vocab_size

def train_bengali_tokenizer(bn_texts, vocab_size, max_length):
    """train bengali bpe tokenizer - runs in own process"""
    print(f"[PID {os.getpid()}] bengali tokenizer training started.")
    print("="*60)
    
    bn_tokenizer = IndicBPETokenizer(vocab_size=vocab_size, max_len=max_length)
    bn_vocab_size = bn_tokenizer.train_bpe(bn_texts)

    # save to disk
    bn_tokenizer_path = os.path.join("tokenizers", f"bengali_bpe_{vocab_size}.pkl")
    bn_tokenizer.save_tokenizer(bn_tokenizer_path)

    print(f"[Bengali] training done! final vocab size: {bn_vocab_size}")
    return bn_vocab_size

if __name__ == '__main__':
    # load up the data first
    print("loading translation data...")
    en_hi_texts, hi_texts = load_translation_data('data/processed/eng2hindi_train.jsonl')
    print(f"got {len(en_hi_texts)} english-hindi pairs")

    en_bn_texts, bn_texts = load_translation_data('data/processed/eng2bengali_train.jsonl')
    print(f"got {len(en_bn_texts)} english-bengali pairs")

    # combine all english texts together for shared tokenizer
    all_english_texts = en_hi_texts + en_bn_texts
    print(f"total english texts: {len(all_english_texts)}")

    # set up parameters
    VOCAB_SIZE = 30_000
    MAX_LENGTH = 1024

    
    print("STARTING PARALLEL TRAINING")
    start_time = time.time()

    # create the process pool and submit jobs
    with mp.Pool(processes=3) as pool:
        # submit all three training jobs at once
        english_job = pool.apply_async(
            train_english_tokenizer, 
            (all_english_texts, VOCAB_SIZE, MAX_LENGTH)
        )
        
        hindi_job = pool.apply_async(
            train_hindi_tokenizer, 
            (hi_texts, VOCAB_SIZE, MAX_LENGTH)
        )
        
        bengali_job = pool.apply_async(
            train_bengali_tokenizer, 
            (bn_texts, VOCAB_SIZE, MAX_LENGTH)
        )
        
        # wait for everything to finish
        print("all jobs submitted, waiting for completion")
        
        # get results (this blocks until each finishes)
        eng_vocab_size = english_job.get()
        hi_vocab_size = hindi_job.get() 
        bn_vocab_size = bengali_job.get()
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f"english vocab: {eng_vocab_size}")
    print(f"hindi vocab: {hi_vocab_size}")
    print(f"bengali vocab: {bn_vocab_size}")
    print(f"total time: {total_time:.2f} seconds ({total_time/60:.1f} mins)")
    print("\ntokenizers saved to disk successfully")
