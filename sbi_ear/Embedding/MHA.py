import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
# batch_size = 32
# block_size = 512
# n_embedding = 512
# n_blocks = 12
# n_head = 8        
dropout = 0.8
# max_iters = 50000
# learning_rate = 3e-4
# eval_every = 500
# eval_iters = 200
# save_every = 10000

import tokenizers
tokenizer = tokenizers.ByteLevelBPETokenizer()
# tokenizer.train("harry-potter.txt")
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

class Head(nn.Module):
    def __init__(self, head_size, n_embedding, block_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        H = k.shape[-1]

        # compute attention scores (affinities)
        wei = q @ k.transpose(-1, -2) * H**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # weighted aggregation of the values
        v = self.value(x) # (B, T, H)
        h = wei @ v # (B, T, T) @ (B, T, H) -> (B, T, H)

        return h

class MuliHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embedding, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embedding, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([h(x) for h in self.heads], dim=-1)
        # print(h)
        h = self.proj(h)
        h = self.dropout(h)
        return h

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.GELU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedding, n_head, block_size):
        super().__init__()
        head_size = n_embedding // n_head
        
        self.sa = MuliHeadAttention(n_head, head_size, n_embedding, block_size)
        self.ff = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        h = x       
        h = h + self.sa(self.ln1(h))
        h = h + self.ff(self.ln2(h))
        return h


class GPTLanguageModel(nn.Module):
    def __init__(self, n_embedding, n_head, n_blocks, block_size, device = "cpu"):
        super().__init__()
        # self.maxpool = nn.AvgPool1d(128, stride=16)
        self.g  = nn.Linear(2, 32) 
        self.e1 = nn.ELU()
        self.f = nn.Linear(32, 128)
        self.e2 = nn.ELU()

        self.e = nn.Flatten()
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.l1 = nn.Linear(48256,12000)
        self.l2 = nn.Linear(12000, 6000)
        self.l3 = nn.Linear(6000, 1000)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head, block_size) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.lm_head = nn.Linear(n_embedding, vocab_size)

        self.device = device
        
    def forward(self, x): # (B, T)
        # h = x  #(B, 2, 6144) L = 6144
        # x = torch.permute(x, (0, 2, 1)) #(B, 6144, 2)
        # print(x.is_cuda, 'a')
        # x = self.e2(self.f(self.e1(self.g(x)))) #(B, L, 128)
        # print(x.is_cuda, 'b', np.shape(x), np.shape(x.long()))

        B, T = x.shape # B, 6144
        tok_emb = self.token_embedding_table(x.long()) # (B, T, C)
        print(tok_emb.size())
        pos_emb = self.position_embedding_table(torch.arange(T, device= self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x.double()) # B, 6144, C
        x = self.ln_f(x) # B, 6144, C

        x = torch.permute(x, (0, 2, 1)) # B, C, 6144
        x = self.maxpool(x)  #(B, 128, 377) L' = 377

        x = self.e(x) #(B, 48256)
        x = self.e1(self.l3(self.e1(self.l2(self.e1(self.l1(x)))))) #(B, 1000)

        return x

class MultiHeadAttentionwithMLP(nn.Module):
    def __init__(self,n_embedding, n_head, n_blocks, block_size):
        super().__init__()
        
        self.maxpool = nn.AvgPool1d(128, stride=16)
        self.g  = nn.Linear(2, 32) 
        self.e1 = nn.ELU()
        self.f = nn.Linear(32, 128)
        self.e2 = nn.ELU()
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head, block_size) for _ in range(n_blocks)])
        self.e = nn.Flatten()
        self.ln_f = nn.LayerNorm(n_embedding) # final layer norm
        self.l1 = nn.Linear(48256,12000)
        self.l2 = nn.Linear(12000, 6000)
        self.l3 = nn.Linear(6000, 1000)

    def forward(self, x):
        h = x  #(B, 2, 6144) L = 6144
        # print('1', h.size()) 
        h = self.maxpool(h)  #(B, 2, 377) L' = 377
        # print('2', h.size())
        h = torch.permute(h, (0, 2, 1)) #(B, L', 2)
        # print('3', h.size())
        h = self.e2(self.f(self.e1(self.g(h)))) #(B, L', 128)
        # print('4', h.size())
        h = self.blocks(h) #(B, L', 128)
        h = self.ln_f(h)
        h = self.e(h) #(B, 48256)
        h = self.e1(self.l3(self.e1(self.l2(self.e1(self.l1(h)))))) #(B, 1000)
        
        # print('5', h.size())
        # h = self.e(h)    #(B, L', 1)
        # print('6', h.size())
        # h = torch.squeeze(h)   #(B, L')
        # print('7', h.size())

        ## h = h.to(torch.double)

        return h
    

    import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################################################################
class SpectralTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_bins, output_dim):
        super(SpectralTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        
        # Embedding layer for spectral bins (if you use discrete spectral indices)
        self.embedding = nn.Embedding(num_bins, embedding_dim)
        
        # Positional encoding parameter (for the training sequence length)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_bins, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(embedding_dim, output_dim)

    def interpolate_positional_encoding(self, input_length):
        """
        Interpolate positional encoding to match input sequence length.
        """
        original_length = self.positional_encoding.size(1)  # The length used during training (e.g., 1298)
        
        if input_length == original_length:
            # No need for interpolation if input length matches training length
            return self.positional_encoding
        else:
            # Interpolate the positional encoding to match input_length
            interpolated_pos_enc = F.interpolate(
                self.positional_encoding.transpose(1, 2),  # (batch, dim, seq_len)
                size=(input_length),  # New sequence length (e.g., 1296)
                mode="linear",
                align_corners=True
            ).transpose(1, 2)  # Back to (batch, seq_len, dim)
            return interpolated_pos_enc

    def forward(self, spectral_data):
        batch_size, seq_length = spectral_data.size(0), spectral_data.size(1)
        
        # Embed the spectral data (use bin index directly or process via embedding)
        embedded_data = self.embedding(spectral_data)
        
        # Dynamically adjust positional encoding to match the input sequence length
        pos_enc = self.interpolate_positional_encoding(seq_length)
        
        # Add positional encoding to the embeddings
        embedded_data += pos_enc[:, :seq_length, :].expand(batch_size, -1, -1)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(embedded_data)
        
        # Aggregate transformer output (e.g., using mean pooling)
        output = torch.mean(transformer_output, dim=1)
        
        # Final classification layer
        output = self.fc(output)
        return output

# Example usage
# Assume your model is trained with 1298 bins but test data has 1296 bins
num_bins_train = 1298
num_bins_test = 1296
embedding_dim = 64
num_heads = 8
num_layers = 6
output_dim = 10

# Instantiate the model
model = SpectralTransformer(embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, num_bins=num_bins_train, output_dim=output_dim)

# Example test with spectral data having 1296 bins (while the model was trained on 1298 bins)
test_spectral_data = torch.randint(0, num_bins_train, (32, num_bins_test))  # Batch of 32 samples, each with 1296 bins
output = model(test_spectral_data)

print(output.shape)  # Output shape should be (32, output_dim)

        
#################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.pe[:, :x.size(1)]
        return x

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, C, latent_dim, num_heads=4):
        super().__init__()
        self.conv = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(C)
        self.bn2 = nn.BatchNorm1d(C)
        self.relu = nn.ReLU()

        self.pe = PositionalEncoding(C)

        encoder_layer = nn.TransformerEncoderLayer(d_model=C, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.linear = nn.Linear(C, latent_dim)

    def forward(self, x, src_key_padding_mask=None):  # x: (B, T, C)
        x = x.transpose(1, 2)               # (B, C, T)
        x = self.conv(x)                    # (B, C, T)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = x.transpose(1, 2)               # (B, T, C)
        
        x = self.pe(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, C)
        z = self.linear(x)                  # (B, T, latent_dim)
        return z

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, C, latent_dim, num_heads=4):
        super().__init__()
        self.pe = PositionalEncoding(C)
        self.cross_attn = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads, batch_first=True)
        
        self.linear1 = nn.Linear(C, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, C)

    def forward(self, x, z, src_key_padding_mask=None):
        # x: (B, T, C), z: (B, T, latent_dim)
        x = self.pe(x)

        # Apply cross-attention with mask support
        attn_out, _ = self.cross_attn(x, z, z, key_padding_mask=src_key_padding_mask)
        
        x = self.linear1(attn_out)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = self.linear3(x)  # same shape as input spectra
        return out

# --- Full Model with Random Masking ---
class TransformerAutoencoder(nn.Module):
    def __init__(self, C, latent_dim, num_heads=4, mask_ratio=0.15):
        super().__init__()
        self.encoder = Encoder(C, latent_dim, num_heads)
        self.decoder = Decoder(C, latent_dim, num_heads)
        self.mask_ratio = mask_ratio

    def generate_random_mask(self, B, T, device):
        num_mask = int(T * self.mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            mask_idx = torch.randperm(T, device=device)[:num_mask]
            mask[b, mask_idx] = True
        return mask

    def forward(self, spectra, mask=None, apply_random_mask=False):
        B, T, _ = spectra.shape

        if apply_random_mask:
            rand_mask = self.generate_random_mask(B, T, spectra.device)
            mask = rand_mask if mask is None else mask | rand_mask

        z = self.encoder(spectra, src_key_padding_mask=mask)
        out = self.decoder(spectra, z, src_key_padding_mask=mask)
        return out, z, mask

# --- Example usage ---
B, T, C, latent_dim = 8, 50, 64, 32
spectra = torch.randn(B, T, C)

model = TransformerAutoencoder(C=C, latent_dim=latent_dim, mask_ratio=0.2)
output, z, mask = model(spectra, apply_random_mask=True)
print("Output shape:", output.shape)  # (B, T, C)
print("Latent shape:", z.shape)      # (B, T, latent_dim)
print("Mask shape:", mask.shape)     # (B, T)
