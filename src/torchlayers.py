import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerMT(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 n_heads=8, 
                 n_layers=6, 
                 d_ff=2048, 
                 max_seq_len=256, 
                 dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = nn.Parameter(self._create_positional_encoding(max_seq_len, d_model))

        # transformer layers using pytorch built-ins
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # important: use batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # important: use batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        self._init_weights()

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq, pad_token=0):
        # pytorch transformer expects True for positions to IGNORE
        return seq == pad_token

    def create_causal_mask(self, size, device):
        # pytorch transformer expects True for positions to IGNORE
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    def forward(self, src, tgt):
        # create masks
        src_key_padding_mask = self.create_padding_mask(src)
        tgt_key_padding_mask = self.create_padding_mask(tgt)
        
        # causal mask for decoder (prevent looking ahead)
        seq_len = tgt.size(1)
        tgt_mask = self.create_causal_mask(seq_len, tgt.device)

        # embeddings with positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:src_emb.size(1)].unsqueeze(0)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:tgt_emb.size(1)].unsqueeze(0)
        tgt_emb = self.dropout(tgt_emb)

        # encode
        encoder_output = self.encoder(
            src_emb, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # decode
        decoder_output = self.decoder(
            tgt_emb, 
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # project to vocab
        output = self.output_projection(decoder_output)
        return output

    def generate(self, src, max_length=50, temperature=0.0):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # encode source
            src_key_padding_mask = self.create_padding_mask(src)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = src_emb + self.pos_encoding[:src_emb.size(1)].unsqueeze(0)

            encoder_output = self.encoder(
                src_emb, 
                src_key_padding_mask=src_key_padding_mask
            )
            
            # start with SOS token
            generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            
            for step in range(max_length):
                # create target embeddings
                tgt_emb = self.tgt_embedding(generated) * math.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoding[:generated.size(1)].unsqueeze(0)
                
                # create masks
                seq_len = generated.size(1)
                tgt_mask = self.create_causal_mask(seq_len, device)
                
                # decode
                decoder_output = self.decoder(
                    tgt_emb,
                    encoder_output,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # get next token logits
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # add EOS penalty for early steps to prevent premature stopping
                if step < 3:
                    logits[:, 3] -= 5.0  # penalize EOS token
                
                if temperature == 0.0:  # greedy decoding
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:  # sampling
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # stop if all sequences generated EOS (after minimum length)
                if step > 2 and (next_token == 3).all():
                    break
            
            return generated[:, 1:]  # remove SOS token

    def generatev2(self, 
                   src, 
                   max_length=50, 
                   temperature=0.0, 
                   min_length=20, 
                   repetition_penalty=1.1, 
                   length_penalty=0.8):

        self.eval()
        
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # encode source once
            src_key_padding_mask = self.create_padding_mask(src)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = src_emb + self.pos_encoding[:src_emb.size(1)].unsqueeze(0)
            
            # forward through enc
            encoder_output = self.encoder(src_emb,  src_key_padding_mask=src_key_padding_mask)
            
            # initialize output
            generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # track tokens for repetition penalty
            vocab_size = self.output_projection.out_features
            token_counts = torch.zeros(batch_size, vocab_size, device=device)

            for step in range(max_length):
                if finished.all():
                    break

                # create tgt embeddings and decode
                tgt_emb = self.tgt_embedding(generated) * math.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoding[:generated.size(1)].unsqueeze(0)
                tgt_mask = self.create_causal_mask(generated.size(1), device)
                
                decoder_output = self.decoder(tgt_emb, 
                                              encoder_output, 
                                              tgt_mask=tgt_mask,
                                              memory_key_padding_mask=src_key_padding_mask)
                logits = self.output_projection(decoder_output[:, -1, :])

                # apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        if not finished[i]:
                            for token in generated[i]:
                                if token_counts[i, token] > 0:
                                    logits[i, token] /= repetition_penalty
                
                # prevent early EOS
                if step < min_length:
                    logits[:, 3] = -float('inf')
                
                # apply length penalty
                if step > min_length and length_penalty != 1.0:
                    logits[:, 3] += math.log(length_penalty)

                # generate greedy token 
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # temp scaling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                    
                    # mantain 90% probability mass
                    cutoff_mask = cumulative_probs > 0.9
                    cutoff_mask[..., 0] = False  # at least one token

                    sorted_logits[cutoff_mask] = -float('inf')
                    logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
                    
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                # update token counts
                for i in range(batch_size):
                    if not finished[i]:
                        token_counts[i, next_token[i, 0]] += 1
                
                generated = torch.cat([generated, next_token], dim=1)
                finished |= (next_token.squeeze(-1) == 3) & (step >= min_length)
            
            return generated[:, 1:]
