#!/usr/bin/env python
# coding: utf-8

import json
import os
from dataclasses import dataclass
import datetime
import math

import torch
import wandb
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.tokenizers.eng import EnglishTokenizer      # simple word-level tokenizer
from src.tokenizers.indic import IndicTokenizer      # simple word-level tokenizer
from src.utils import simple_train_val_split
from src.dataloader import DataLoaderLite
from src.components import TransformerMT  as selftx
from src.torchlayers import TransformerMT as torchtx
from src.scores import calculate_translation_metrics

# wandb params
wandb_entity        = "vasudev-gupta-decision-tree-analytics-services"
wandb_project       = "iitk-nlp-capstone"
wandb_experiment    = "exp5-eng-bengali-transformer-built-in"

@dataclass
class TransformerConfig:
   SRC_VOCAB_SIZE: int = 30_000                      # source vocabulary size
   TGT_VOCAB_SIZE: int = 30_000                      # target vocabulary size
   SRC_MAX_LENGTH: int = 256                         # max sequence length source lang
   TGT_MAX_LENGTH: int = 256                         # max sequence length target lang
   D_MODEL: int = 128                                # embedding dimension
   N_HEADS: int = 4                                  # number of heads in attention
   N_LAYERS: int = 6                                 # number of transformer blocks
   D_FF: int = 128 * 4                               # dimension of feedforward (4x of embedding dims)
   MAX_SEQ_LEN: int = 256
   DROPOUT: float = 0.1
   BATCH_SIZE: int = 32
   EVAL_STEPS: int = 100
   EPOCHS: int = 10


config = TransformerConfig()

en_texts = []
be_texts = []

with open('data/processed/eng2bengali_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        en_texts.append(data['source'])
        be_texts.append(data['target'])

print(f"loaded {len(en_texts)} english-bengali pairs")

train_indices, val_indices = train_test_split(range(len(en_texts)), 
                                              test_size=0.1, 
                                              random_state=42)
# old tokenizers - word level
# eng tokenizer
eng_tokenizer = EnglishTokenizer(max_vocab_size=config.SRC_VOCAB_SIZE, max_length=config.SRC_MAX_LENGTH)
eng_tokenizer.build_vocab(en_texts)

# indic tokenizer
indic_tokenizer = IndicTokenizer(max_vocab_size=config.SRC_VOCAB_SIZE, max_length=config.SRC_MAX_LENGTH)
indic_tokenizer.build_vocab(be_texts)

en_sequences = eng_tokenizer.texts_to_sequences(en_texts)
be_sequences = indic_tokenizer.texts_to_sequences(be_texts)

# Split texts
en_train_texts = [en_texts[i] for i in train_indices]
en_val_texts   = [en_texts[i] for i in val_indices]
be_train_texts = [be_texts[i] for i in train_indices]
be_val_texts   = [be_texts[i] for i in val_indices]

# Split sequences
en_train = en_sequences[train_indices]
en_val   = en_sequences[val_indices]
be_train = be_sequences[train_indices]
be_val   = be_sequences[val_indices]

# split the data using your function
# en_train, en_val, be_train, be_val = simple_train_val_split(en_sequences, be_sequences, test_size=0.1, random_state=42)

# convert to torch tensors
en_train_tensor = torch.tensor(en_train, dtype=torch.long)
be_train_tensor = torch.tensor(be_train, dtype=torch.long)
en_val_tensor = torch.tensor(en_val,     dtype=torch.long)
be_val_tensor = torch.tensor(be_val,     dtype=torch.long)

train_loader = DataLoaderLite(X=en_train_tensor, y=be_train_tensor, batch_size=config.BATCH_SIZE)
val_loader = DataLoaderLite(X=en_val_tensor,     y=be_val_tensor,   batch_size=config.BATCH_SIZE)

# MODEL
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Transformer Model
device = get_device()
# transformer_model = selftx(src_vocab_size=eng_tokenizer.vocab_size,
#                            tgt_vocab_size=eng_tokenizer.vocab_size,
#                            d_model=config.D_MODEL,
#                            n_heads=config.N_HEADS,
#                            n_layers=config.N_LAYERS,
#                            d_ff=config.D_FF,
#                            max_seq_len=config.MAX_SEQ_LEN,
#                            dropout=config.DROPOUT)
transformer_model = torchtx(src_vocab_size=eng_tokenizer.vocab_size,
                            tgt_vocab_size=indic_tokenizer.vocab_size,
                            d_model=config.D_MODEL,
                            n_heads=config.N_HEADS,
                            n_layers=config.N_LAYERS,
                            d_ff=config.D_FF,
                            max_seq_len=config.MAX_SEQ_LEN,
                            dropout=config.DROPOUT)
transformer_model.to(device)

# Loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0)          # ignore padding tokens

# Steps in optimisation
MAX_STEPS = int((en_train_tensor.shape[0] / config.BATCH_SIZE) * config.EPOCHS)

# LR Scheduler params
max_lr        = 6e-4                        # Peak learning rate after warmup, used in cosine decay schedule
min_lr        = max_lr * 0.1                # Final learning rate at the end of cosine decay, seen 1/10 lower than peak in other implementations
warmup_steps  = int(MAX_STEPS * 0.035)      # Number of steps for linear LR warmup (3.5% of total training), gradually increases LR from 0 to max_lr

# Optimizer
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=max_lr)
total_params = sum(p.numel() for p in transformer_model.parameters())
print(f"transformer has {total_params:,} parameters")

# Learning Rate
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (MAX_STEPS - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# WandB
wandbrun = wandb.init(entity=wandb_entity,
                      project=wandb_project,
                      name=wandb_experiment,
                      config={
                                "max_seq_len": config.MAX_SEQ_LEN,
                                "steps": MAX_STEPS,
                                "epochs": config.EPOCHS,
                                "src_vocab_size": config.SRC_VOCAB_SIZE,
                                "tgt_vocab_size": config.TGT_VOCAB_SIZE,
                                "d_model": config.D_MODEL,
                                "n_heads": config.N_HEADS,
                                "n_layers": config.N_LAYERS,
                                "d_ff": config.D_FF,
                                "dropout": config.DROPOUT,
                                "batch_size": config.BATCH_SIZE,
                                "eval_steps": config.EVAL_STEPS,
                                "max_lr": max_lr,
                                "min_lr": min_lr,
                                "warmup_steps": warmup_steps,
                                "architecture": "transformer",
                                "task": "english-hindi-translation",
                                "machine": "home-gpu",
                                "device": device,
                                "dataset": "eng2bengali_train.jsonl",
                                "total_params": total_params
                            }
                        )

# setup logging
file = f"logs/training_log_transformer_mt_{wandb_experiment}{datetime.datetime.now().strftime("%Y-%m-%d")}.txt"
if os.path.isfile(file):
    os.remove(file)

logfile = open(file, "w")
logfile.write("Time : {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d")))
logfile.write("total  pramas : {}\n".format(total_params))
logfile.write("Total  Steps  : {}\n".format(MAX_STEPS))
logfile.write("SRC_VOCAB_SIZE: {}\n".format(config.SRC_VOCAB_SIZE))
logfile.write("TGT_VOCAB_SIZE: {}\n".format(config.TGT_VOCAB_SIZE))
logfile.write("D_MODEL       : {}\n".format(config.D_MODEL))
logfile.write("N_HEADS       : {}\n".format(config.N_HEADS))
logfile.write("N_LAYERS      : {}\n".format(config.N_LAYERS))
logfile.write("D_FF          : {}\n".format(config.D_FF))
logfile.write("DROPOUT       : {}\n".format(config.DROPOUT))
logfile.write("BATCH_SIZE    : {}\n".format(config.BATCH_SIZE))
logfile.write("EVAL_STEPS    : {}\n".format(config.EVAL_STEPS))
logfile.write("EPOCHS        : {}\n".format(config.EPOCHS))
logfile.write(f"Training on {device}...\n")
logfile.write("\n=======================\n\n")

print("Step  |  Epoch |  Train Loss  |  Val Loss")
logfile.write("Step  |  Epoch |  Train Loss  |  Val Loss\n")
logfile.write("-" * 70 + "\n")
print("-" * 62)


# Training loop
checkpoint_path = os.path.join("checkpoints", "eng_bengali", wandb_experiment) 
os.makedirs(checkpoint_path, exist_ok=True)
tr_lossi, val_lossi = [], []
avg_val_loss = None

for step in range(1, MAX_STEPS + 1):

    # TRAIN MODE
    transformer_model.train()
    optimizer.zero_grad()

    src_batch, tgt_batch = train_loader.next_batch()
    src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
    training_epoch = train_loader.epoch + 1

    # forward pass - use tgt[:-1] as input, tgt[1:] as target
    tgt_input = tgt_batch[:, :-1]  # exclude last token for input
    tgt_target = tgt_batch[:, 1:]  # exclude first token for target

    outputs = transformer_model(src_batch, tgt_input)

    # reshape for loss calculation
    batch_size, seq_len, vocab_size = outputs.shape
    outputs = outputs.reshape(-1, vocab_size)
    tgt_target = tgt_target.reshape(-1) 

    loss = loss_fn(outputs, tgt_target)

    # backward
    loss.backward()
    tr_lossi.append((step, loss.item()))

    # clip norm of the entire gradient vector to 1.0 after backward (as suggested in gpt3) -> check for its stability during the optimisation
    norm = torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)

    # optimizer step with learning scheduler
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # validation
    if step % config.EVAL_STEPS == 0:
        transformer_model.eval()
        val_loss_sum = 0.0
        val_steps = 200
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = val_loader.next_batch()
                xb, yb = xb.to(device), yb.to(device)

                yb_input = yb[:, :-1]
                yb_target = yb[:, 1:]

                out = transformer_model(xb, yb_input)
                out = out.reshape(-1, out.size(-1))
                yb_target = yb_target.reshape(-1)

                val_loss_sum += loss_fn(out, yb_target).item()

                preds = out.argmax(1)
                non_pad_mask = (yb_target != 0)

        avg_val_loss = val_loss_sum / val_steps
        val_lossi.append((step, avg_val_loss))
        
        # calculate translatoin metrics
        metrics = calculate_translation_metrics(transformer_model, 
                                                eng_tokenizer, 
                                                indic_tokenizer,
                                                en_val_texts, 
                                                be_val_texts, 
                                                device=device, 
                                                num_samples=val_steps)
        avg_val_chrf, avg_val_bleu = metrics['chrf'], metrics['bleu']

        msg = (f"{step:5d} | {training_epoch:5d} | {loss.item():11.4f} | {avg_val_loss:9.4f} | {avg_val_chrf: 8.4f} | {avg_val_bleu: 8.4f}")
    else:
        msg = f"{step:5d}  | {training_epoch:5d}  | {loss.item():11.4f}  |    ----"

    # save checkpoint every 500 steps
    if step % 500 == 0:
        checkpoint = {
            'step': step,
            'epoch': training_epoch,
            'model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'config': config,
            'tr_lossi': tr_lossi,
            'val_lossi': val_lossi,

            # COMPLETE tokenizer data for inference
            'eng_tokenizer': {
                'word2idx': eng_tokenizer.word2idx,
                'idx2word': eng_tokenizer.idx2word,
                'vocab_size': eng_tokenizer.vocab_size,
                'max_vocab_size': eng_tokenizer.max_vocab_size,
                'max_length': eng_tokenizer.max_length,
            },
            'indic_tokenizer': {
                'word2idx': indic_tokenizer.word2idx,
                'idx2word': indic_tokenizer.idx2word,
                'vocab_size': indic_tokenizer.vocab_size,
                'max_vocab_size': indic_tokenizer.max_vocab_size,
                'max_length': indic_tokenizer.max_length,
            },

            # MODEL metadata for reconstruction
            'model_config': {                                    
                'src_vocab_size': config.SRC_VOCAB_SIZE,
                'tgt_vocab_size': config.TGT_VOCAB_SIZE,
                'd_model': config.D_MODEL,
                'n_heads': config.N_HEADS,
                'n_layers': config.N_LAYERS,
                'd_ff': config.D_FF,
                'max_seq_len': config.MAX_SEQ_LEN,
                'dropout': config.DROPOUT
            }
        }

        checkpoint_file = os.path.join(checkpoint_path, f"tx_epoch_{training_epoch}_step_{step}.pt")
        torch.save(checkpoint, checkpoint_file)
        print(f"saved checkpoint at step {step}")
        logfile.write(f"saved checkpoint at step {step}\n")

    # wandb logging
    wandbrun.log({"step": step, 
                    "epoch": training_epoch,
                    "loss": loss.item(), 
                    "val_loss": avg_val_loss,
                    "val_bleu": avg_val_bleu,
                    "val_chrf": avg_val_chrf,
                    "lr": lr, 
                    "norm": norm})

    print(msg)
    logfile.write(msg + "\n")


logfile.close()
print("training completed!")
