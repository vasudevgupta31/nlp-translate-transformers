import re
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm


class EnglishTokenizer:
    # simple english tokenizer 
    
    def __init__(self, max_vocab_size=30_000, max_length=256):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {}                   # word to index
        self.idx2word = {}                   # index to word
        self.vocab_size = 0

    def build_vocab(self, texts):
        # build vocabulary from texts
        # count all words
        word_counts = Counter()
        for text in tqdm(texts):
            words = self.tokenize_text(text)
            word_counts.update(words)

        # add special tokens first
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

        # add most frequent words
        most_common = word_counts.most_common(self.max_vocab_size - 4)  # save space for special tokens
        for word, count in tqdm(most_common):
            self.word2idx[word] = len(self.word2idx)

        # itos # stoi
        # create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        return self.vocab_size

    def tokenize_text(self, text):
        # tokenize english text
        text = text.lower().strip()
        # put spaces around punctuation so they become separate tokens
        text = re.sub(r"([.!?,:;\"'])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        return text.split()

    def texts_to_sequences(self, texts, add_special_tokens=True):
        # convert texts to sequences of indices
        sequences = []

        for text in texts:
            words = self.tokenize_text(text)
            sequence = []
            
            # add start token
            if add_special_tokens:
                sequence.append(self.word2idx['<SOS>'])
            
            # convert words to numbers
            for word in words:
                if len(sequence) >= self.max_length - 1:  # keep space for end token
                    break
                if word in self.word2idx:
                    sequence.append(self.word2idx[word])
                else:
                    sequence.append(self.word2idx['<UNK>'])
            
            # add end token
            if add_special_tokens and len(sequence) < self.max_length:
                sequence.append(self.word2idx['<EOS>'])
            
            # pad to max length
            while len(sequence) < self.max_length:
                sequence.append(self.word2idx['<PAD>'])
            
            sequences.append(sequence)
        
        return np.array(sequences)

    def sequences_to_texts(self, sequences):
        # convert sequences back to readable text
        texts = []
        for sequence in sequences:
            words = []
            for idx in sequence:
                word = self.idx2word.get(idx, '<UNK>')
                if word == '<EOS>':  # stop at end token
                    break
                if word not in ['<PAD>', '<SOS>']:  # skip special tokens
                    words.append(word)
            
            # join words and fix punctuation spacing
            text = ' '.join(words)
            text = re.sub(r' ([.!?,:;])', r'\1', text)  # remove space before punctuation
            texts.append(text)
        
        return texts


    def save(self, filepath):
        # save tokenizer to file
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'max_length': self.max_length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"saved english tokenizer to {filepath}")
    
    def load(self, filepath):
        # load tokenizer from file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        self.max_vocab_size = data['max_vocab_size']
        self.max_length = data['max_length']
        print(f"loaded english tokenizer from {filepath}")
        return self
