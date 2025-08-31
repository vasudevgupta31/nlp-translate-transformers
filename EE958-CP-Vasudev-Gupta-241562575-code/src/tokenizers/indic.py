import re
import pickle
import string
import numpy as np
from collections import Counter

triv_tokenizer_indic_pat=re.compile(r'(['+string.punctuation+r'\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F'+r'])')
pat_num_seq=re.compile(r'([0-9]+ [,.:/] )+[0-9]+')  # date, numbers, section/article numbering

# detokenizer patterns 
left_attach=r'!%)\]},.:;>?\u0964\u0965'
pat_la=re.compile(r'[ ](['+left_attach+r'])')
lr_attach=r'-/\\'
pat_lra=re.compile(r'[ ](['+lr_attach+r'])[ ]')
right_attach=r'#$(\[{<@'
pat_ra=re.compile(r'(['+right_attach+r'])[ ]')


def trivial_tokenize_indic(text): 
    """tokenize string for Indian language scripts using Brahmi-derived scripts

    A trivial tokenizer which just tokenizes on the punctuation boundaries. 
    This also includes punctuations for the Indian language scripts (the 
    purna virama and the deergha virama). This is a language independent 
    tokenizer

    Args:
        text (str): text to tokenize

    Returns:
        list: list of tokens

    """
    tok_str=triv_tokenizer_indic_pat.sub(r' \1 ',text.replace('\t',' '))
    s=re.sub(r'[ ]+',' ',tok_str).strip(' ')

    # do not tokenize numbers and dates
    new_s=''
    prev=0
    for m in pat_num_seq.finditer(s):
        start=m.start()
        end=m.end()
        if start>prev:
            new_s=new_s+s[prev:start]
            new_s=new_s+s[start:end].replace(' ','')
            prev=end
   
    new_s=new_s+s[prev:]
    s=new_s    
    return s.split(' ')


def trivial_detokenize_indic(text): 
    """detokenize string for Indian language scripts using Brahmi-derived scripts

    A trivial detokenizer which:

        - decides whether punctuation attaches to left/right or both
        - handles number sequences
        - handles quotes smartly (deciding left or right attachment)

    Args:
        text (str): tokenized text to process 

    Returns:
        str: detokenized string
    """

    s=text
    ### some normalizations 

    #numbers and dates
    new_s=''
    prev=0
    for m in pat_num_seq.finditer(s):
        start=m.start()
        end=m.end()
        if start>prev:
            new_s=new_s+s[prev:start]
            new_s=new_s+s[start:end].replace(' ','')
            prev=end
   
    new_s=new_s+s[prev:]
    s=new_s

    ###  consective single quotes or backslashes become double quotes
    #s=s.replace("' '", "''")
    #s=s.replace("` `", '``')

    s=pat_lra.sub('\\1',s)
    s=pat_la.sub('\\1',s)
    s=pat_ra.sub('\\1',s)

    # assumes well formedness of quotes and alternates between right and left attach

    alt_attach='\'"`'
    for punc in alt_attach: 
        cnt=0
        out_str=[]
        for c in s:
            if c == punc:
                if cnt%2==0:
                    out_str.append('@RA')
                else:
                    out_str.append('@LA')
                cnt+=1    
            else:
                out_str.append(c)

        s=''.join(out_str).replace('@RA ',punc).replace(' @LA',punc
                ).replace('@RA',punc).replace('@LA',punc)

    return s


class IndicTokenizer:

    # simple indic tokenizer using indicnlp functions

    def __init__(self, max_vocab_size=30_000, max_length=256):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {}                   # word to index
        self.idx2word = {}                   # index to word
        self.vocab_size = 0
        self.trivial_tokenize_indic = trivial_tokenize_indic
        self.trivial_detokenize_indic = trivial_detokenize_indic
        self.indicnlp_available = True

    def build_vocab(self, texts):
        # build vocabulary from texts
        # count all words
        word_counts = Counter()
        for text in texts:
            words = self.tokenize_text(text)
            word_counts.update(words)

        # add special tokens first
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

        # add most frequent words
        most_common = word_counts.most_common(self.max_vocab_size - 4)  # save space for special tokens
        for word, count in most_common:
            self.word2idx[word] = len(self.word2idx)

        # create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        return self.vocab_size

    def tokenize_text(self, text):
        # tokenize indic text using indicnlp
        if self.indicnlp_available:
            try:
                tokens = self.trivial_tokenize_indic(text)
                return [token for token in tokens if token.strip()]  # remove empty tokens
            except Exception as e:
                print(f"indicnlp tokenization failed: {e}")
        
        # fallback if indicnlp doesn't work
        text = text.strip()
        text = re.sub(r'([।॥.!?,:;])', r' \1 ', text)  # space around indic punctuation
        text = re.sub(r'\s+', ' ', text)
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
                if word not in ['<PAD>', '<SOS>', '<UNK>']:  # skip special tokens and unknown
                    words.append(word)
            
            # Only process if we have actual words
            if not words:
                texts.append("")
                continue
            
            # use indicnlp for proper detokenization
            if self.indicnlp_available and words:
                try:
                    tokenized_text = ' '.join(words)
                    text = self.trivial_detokenize_indic(tokenized_text)
                    # Additional cleanup to remove any remaining special tokens
                    text = text.replace('<EOS>', '').replace('<SOS>', '').replace('<PAD>', '').replace('<UNK>', '')
                    text = re.sub(r'\s+', ' ', text).strip()  # clean up extra spaces
                    texts.append(text)
                    continue
                except Exception as e:
                    print(f"indicnlp detokenization failed: {e}")
            
            # fallback detokenization
            text = ' '.join(words)
            text = re.sub(r' ([।॥.!?,:;])', r'\1', text)  # remove space before punctuation
            # Additional cleanup
            text = text.replace('<EOS>', '').replace('<SOS>', '').replace('<PAD>', '').replace('<UNK>', '')
            text = re.sub(r'\s+', ' ', text).strip()  # clean up extra spaces
            texts.append(text)

        return texts

    def save(self, filepath):
        # save tokenizer to file
        data = {'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'max_vocab_size': self.max_vocab_size,
                'max_length': self.max_length}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"saved indic tokenizer to {filepath}")

    def load(self, filepath):
        # load tokenizer from file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        self.max_vocab_size = data['max_vocab_size']
        self.max_length = data['max_length']
        print(f"loaded indic tokenizer from {filepath}")
        return self
