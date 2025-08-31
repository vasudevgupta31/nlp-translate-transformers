import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as proposed in the original Transformer paper."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))             # equivalent to (10000^(2i/d_model)) but better for numerical stability

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    # multi-head attention mechanism
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # apply attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # standard 4x expansion (I read in gpt2 paper first time)

        self.c_fc = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU(approximate='tanh')    # inspiration from gpt2/3 paper
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    # single transformer encoder layer
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    # single transformer decoder layer
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # self-attention
        attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # cross-attention with encoder output
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerMT(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=128, 
                 n_heads=4, 
                 n_layers=6, 
                 d_ff=128*4, 
                 max_seq_len=256, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # encoder and decoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) 
                                             for _ in range(n_layers)])
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) 
                                             for _ in range(n_layers)])

        # output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq, pad_token=0):
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0

    def forward(self, src, tgt):
        # create masks
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask

        # encode
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # decode
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)

        # project to vocab
        output = self.output_projection(decoder_output)
        return output

    def generate(self, src, max_length=50, temperature=0.0):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # encode source
            src_mask = self.create_padding_mask(src)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.pos_encoding(src_emb)
            
            encoder_output = src_emb
            for layer in self.encoder_layers:
                encoder_output = layer(encoder_output, src_mask)
            
            # start with SOS token
            generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            
            for step in range(max_length):
                # create target mask
                tgt_mask = self.create_padding_mask(generated)
                seq_len = generated.size(1)
                look_ahead_mask = self.create_look_ahead_mask(seq_len).to(device)
                tgt_mask = tgt_mask & look_ahead_mask
                
                # decode
                tgt_emb = self.tgt_embedding(generated) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoding(tgt_emb)
                
                decoder_output = tgt_emb
                for layer in self.decoder_layers:
                    decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
                
                # get next token logits
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # AVOID predicting EOS too early - add penalty
                logits[:, 3] -= 5.0  # penalize EOS token for first few steps
                
                if temperature == 0.0:  # greedy
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # only stop if we have generated at least 3 tokens AND hit EOS
                if step > 2 and (next_token == 3).all():
                    break
            
            return generated[:, 1:]  # remove SOS token
