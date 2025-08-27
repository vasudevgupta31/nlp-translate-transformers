import re
import pickle
import numpy as np
import string
from collections import Counter, defaultdict
from tqdm import tqdm

# taken from indicnlp
indic_punct_chars = string.punctuation + '\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F'
punct_pattern = re.compile(r'([' + re.escape(indic_punct_chars) + r'])')
number_pattern = re.compile(r'([0-9]+ [,.:/] )+[0-9]+')

# patterns taken from inidicnlp
left_punct = r'!%)\]},.:;>?\u0964\u0965'
left_pattern = re.compile(r' ([' + re.escape(left_punct) + r'])')
both_punct = r'-/\\'
both_pattern = re.compile(r' ([' + re.escape(both_punct) + r']) ')
right_punct = r'#$(\[{<@'
right_pattern = re.compile(r'([' + re.escape(right_punct) + r']) ')


def tokenize_indic_text(text):
    """tokenization for indic text"""
    if not text or not text.strip():
        return []
    text = text.strip()
    text = number_pattern.sub(r' \1 ', text)
    text = punct_pattern.sub(r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.split()


def detokenize_indic_text(text):
    """fix punctuation spacing"""
    if not text:
        return ""
    text = left_pattern.sub(r'\1', text)
    text = both_pattern.sub(r'\1', text)
    text = right_pattern.sub(r'\1', text)
    return text.strip()


def count_pairs(token_list, pair_counts=None):
    """
    count pairs in a single token list
    """
    if pair_counts is None:
        pair_counts = defaultdict(int)

    for i in range(len(token_list) - 1):
        pair = (token_list[i], token_list[i+1])
        pair_counts[pair] += 1
    return pair_counts


def merge_tokens(token_list, target_pair, new_token_id):
    """
    replace all occurences of target_pair with new_token_id
    so [1,2,3,1,2] with pair (1,2) becomes [4,3,4] if new_token_id is 4
    """
    result = []
    i = 0
    while i < len(token_list):
        # check if current position matches our target pair
        if (i < len(token_list) - 1 and 
            token_list[i] == target_pair[0] and 
            token_list[i+1] == target_pair[1]):
            result.append(new_token_id)
            i += 2  # skip both tokens since we merged them
        else:
            result.append(token_list[i])
            i += 1
    return result


class IndicBPETokenizer:
    
    def __init__(self, vocab_size=30000, max_len=256):
        self.vocab_size = vocab_size
        self.max_len = max_len

        # INDIC REGEX PATTERN
        # - Indic script characters
        # - Numbers 
        # - Punctuation including Indic punctuation
        # - Spaces (preserved as separate tokens)
        # - English characters mixed in
        indic_chars = r'[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF]'
        english_chars = r'[a-zA-Z]'
        numbers = r'[0-9]'
        indic_punct = r'[\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F]'
        regular_punct = r'[' + re.escape(string.punctuation) + r']'

        # include spaces as separate tokens
        self.pattern = f"""({indic_chars}+|{english_chars}+|{numbers}+|{indic_punct}|{regular_punct}|\\s+)"""
        self.regex_splitter = re.compile(self.pattern)
        
        self.merges = {}  # (token1, token2) -> new_token_id
        self.vocab = {}   # token_id -> actual bytes
        self.special_tokens = {'<PAD>': 256,   #  padding sequences  
                               '<UNK>': 257,   #  unknown stuff
                               '<EOS>': 258    #  end of sequence
                               }
        self.special_lookup = {v: k for k, v in self.special_tokens.items()}

    def train_bpe(self, text_list, show_progress=False):
        """
        train the bpe tokenizer on a list of texts
        """
        print(f"starting bpe training on {len(text_list)} texts...")

        if self.vocab_size < 259:
            raise ValueError("vocab size too small, need at least 259")

        num_merges = self.vocab_size - 259
        
        # process text using word frequencies
        print("collecting word frequencies...")
        word_freq = Counter()
        
        # proces in large batches
        batch_size = 10000
        for i in tqdm(range(0, len(text_list), batch_size), desc="processing text batches"):
            batch_texts = text_list[i:i+batch_size]
            
            for text in batch_texts:
                if text and text.strip():
                    # use indic regex to split into chunks
                    chunks = re.findall(self.regex_splitter, text)
                    # keep all chunks including spaces, only filter truly empty ones
                    chunks = [chunk for chunk in chunks if chunk]
                    # count frequency of each unique chunk
                    word_freq.update(chunks)

        print(f"found {len(word_freq)} unique text chunks")

        # convert to bytes only once per unique chunk
        print("converting unique chunks to byte sequences...")
        word_to_tokens = {}

        for word, freq in tqdm(word_freq.items(), desc="processing unique chunks"):
            byte_data = word.encode("utf-8")
            token_ids = list(byte_data)
            word_to_tokens[word] = (token_ids, freq)

        # init vocab
        vocab_dict = {}
        for i in range(256):
            vocab_dict[i] = bytes([i])
        
        # BPE merging using word frequencies
        print(f"doing {num_merges} bpe merge operations...")
        merge_dict = {}

        for merge_step in tqdm(range(num_merges), desc="bpe merges"):
            
            # count pairs weighted by word frequency
            all_pairs = defaultdict(int)
            
            for word, (tokens, freq) in word_to_tokens.items():
                if len(tokens) < 2:
                    continue
                    
                # count pairs in this word, weighted by frequency
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    all_pairs[pair] += freq

            # if no pairs left, we're done
            if not all_pairs:
                print(f"no more pairs to merge at step {merge_step}")
                break
            
            # find the most common pair
            best_pair = max(all_pairs.keys(), key=lambda x: all_pairs[x])
            
            # create new token id for this merge
            new_id = 259 + merge_step

            # apply merge only to affected words
            new_word_to_tokens = {}
            for word, (tokens, freq) in word_to_tokens.items():
                new_tokens = merge_tokens(tokens, best_pair, new_id)
                new_word_to_tokens[word] = (new_tokens, freq)
            
            word_to_tokens = new_word_to_tokens
            
            # save this merge
            merge_dict[best_pair] = new_id
            vocab_dict[new_id] = vocab_dict[best_pair[0]] + vocab_dict[best_pair[1]]
            
            if show_progress and merge_step % 500 == 0:
                print(f"step {merge_step+1}: merged {best_pair} -> {new_id} (appeared {all_pairs[best_pair]} times)")
        
        # save everything
        self.merges = merge_dict
        self.vocab = vocab_dict
        
        print(f"training done! final vocab size: {len(self.vocab)}")
        return len(self.vocab)

    def _encode_single_chunk(self, byte_data):
        """
        encode one chunk of bytes using our trained bpe merges
        this is the core encoding logic
        """
        tokens = list(byte_data)
        
        # keep merging until no more merges possible
        while len(tokens) >= 2:
            # count pairs in current token sequence
            pairs = count_pairs(tokens)
            
            # find the pair with the lowest merge index (earliest in training)
            best_pair = None
            best_merge_idx = float('inf')
            
            for pair in pairs:
                if pair in self.merges:
                    if self.merges[pair] < best_merge_idx:
                        best_merge_idx = self.merges[pair]
                        best_pair = pair
            
            # if no valid merges left, stop
            if best_pair is None:
                break
            
            # apply the best merge
            merge_id = self.merges[best_pair]
            tokens = merge_tokens(tokens, best_pair, merge_id)
        
        return tokens

    def encode_text(self, text):
        """
        encode a text string into token ids
        splits text first then encodes each chunk
        """
        # split into chunks using our indic regex
        text_chunks = re.findall(self.regex_splitter, text)
        
        # encode each chunk separately
        all_tokens = []
        for chunk in text_chunks:
            if chunk:  # only skip truly empty chunks, preserve spaces
                chunk_bytes = chunk.encode("utf-8")
                chunk_tokens = self._encode_single_chunk(chunk_bytes)
                all_tokens.extend(chunk_tokens)
        
        return all_tokens

    def decode_tokens(self, token_ids):
        """
        decode a list of token ids back into text
        handles special tokens appropriately
        """
        byte_parts = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            elif token_id in self.special_lookup:
                # skip special tokens in output
                special_tok = self.special_lookup[token_id]
                if special_tok not in ['<PAD>', '<UNK>', '<EOS>']:
                    byte_parts.append(special_tok.encode("utf-8"))
            else:
                # unknown token
                byte_parts.append(b'<UNK>')

        # combine all bytes and decode to string
        full_bytes = b"".join(byte_parts)
        text = full_bytes.decode("utf-8", errors="replace")
        
        # apply indicnlp detokenization
        try:
            text = detokenize_indic_text(text)
        except:
            # fallback detokenization
            text = re.sub(r' ([।॥.!?,:;])', r'\1', text)
        
        return text

    def texts_to_sequences(self, text_list, add_special=True):
        """
        convert list of texts to padded sequences of token ids
        this matches the old interface
        """
        sequences = []
        
        # process in small batches for progress tracking
        batch_size = 100
        for i in tqdm(range(0, len(text_list), batch_size), desc="converting texts"):
            batch = text_list[i:i+batch_size]
            
            for text in batch:
                # encode the text
                token_ids = self.encode_text(text)
                
                sequence = []
                
                # add start token if requested
                if add_special:
                    sequence.append(self.special_tokens['<EOS>'])
                
                # add actual content tokens
                for tok_id in token_ids:
                    if len(sequence) >= self.max_len - 1:  # save space for end token
                        break
                    sequence.append(tok_id)
                
                # add end token if there's room
                if add_special and len(sequence) < self.max_len:
                    sequence.append(self.special_tokens['<EOS>'])
                
                # pad to max length
                pad_token = self.special_tokens['<PAD>']
                while len(sequence) < self.max_len:
                    sequence.append(pad_token)
                
                sequences.append(sequence)
        
        # convert to numpy array for efficiency
        return np.array(sequences, dtype=np.int32)

    def sequences_to_texts(self, sequences):
        """
        convert padded sequences back to readable texts
        removes padding and special tokens appropriately  
        """
        texts = []
        for sequence in sequences:
            # clean up the sequence
            clean_tokens = []
            for token_id in sequence:
                # stop at end token
                if token_id == self.special_tokens['<EOS>']:
                    break
                # skip padding tokens
                if token_id != self.special_tokens['<PAD>']:
                    clean_tokens.append(token_id)
            
            # decode back to text
            text = self.decode_tokens(clean_tokens)
            texts.append(text)
        
        return texts

    def save_tokenizer(self, file_path):
        """save the trained tokenizer"""
        data_to_save = {
            'merges': self.merges,
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'pattern': self.pattern,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"saved tokenizer to {file_path}")

    def load_tokenizer(self, file_path):
        """load a previously trained tokenizer"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # restore all the data
        self.merges = data['merges']
        self.vocab = data['vocab']
        self.special_tokens = data['special_tokens']
        self.pattern = data['pattern']
        self.vocab_size = data['vocab_size']
        self.max_len = data['max_len']
        
        # rebuild derived stuff
        self.regex_splitter = re.compile(self.pattern)
        self.special_lookup = {v: k for k, v in self.special_tokens.items()}
        
        print(f"loaded tokenizer from {file_path}")
        return self

    def get_stats(self):
        """get stats about the tokenizer"""
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'pattern': self.pattern
        }
