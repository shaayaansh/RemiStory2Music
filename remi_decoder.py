import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].detach()
        return x


class RemiDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,     
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.out_proj = nn.Linear(d_model, vocab_size)

    
    def forward(
        self,
        input_ids,
        tgt_key_padding_mask=None,
        memory=None,
    ):
        B, T = input_ids.size() # input: (B, T)
        x = self.token_embedding(input_ids) # (B, T, d_model)
        x = self.pos_encoder(x) # (B, T, d_model)

        # causal mask
        tgt_mask = Transformer.generate_square_subsequent_mask(T).to(input_ids.device)

        if memory is None:
            memory = torch.zeros_like(x)

        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.out_proj(decoded)

        return logits


    def generate(
        self,
        bos_id,
        eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
        memory=None,
    ):
        device = device or torch.device("cpu")
        generated = [bos_id]

        with torch.no_grad():
            for _ in range(1, max_len):
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)  
                logits = self.forward(input_tensor, memory=memory)
                last_logits = logits[0, -1]

                if decoding_strategy == "greedy":
                    next_token = int(last_logits.argmax())

                elif decoding_strategy == "top_p":
                    next_token = int(self.top_p_sample(last_logits, p=top_p))

                else:
                    raise ValueError("Unsupported decoding strategy")

                generated.append(next_token)

                if next_token == eos_id:
                    break

        return generated



    def top_p_sample(self, logits, p=0.9):
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cum_probs <= p
        mask[0] = True  # Always include at least the top token

        filtered_probs = sorted_probs[mask]
        filtered_idx = sorted_idx[mask]

        filtered_probs = filtered_probs / filtered_probs.sum()
        choice = torch.multinomial(filtered_probs, 1)
        return filtered_idx[choice].item()