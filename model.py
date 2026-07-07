import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, nhead, n_layers):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        # final LayerNorm needed because norm_first=True (pre-norm) doesnt normalize the output of the last block
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layers, norm=nn.LayerNorm(d_model))
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len, device=x.device)
        x = self.transformer(x, causal_mask)

        logits = self.linear(x)
        return logits