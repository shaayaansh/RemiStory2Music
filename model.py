import torch
import torch.nn as nn
from transformers import AutoModel
from remi_decoder import RemiDecoder, PositionalEncoding
import torch.nn.functional as F


class Story2MusicTransformer(nn.Module):
    def __init__(
        self,
        encoder_name,
        decoder
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.decoder = decoder
        self.memory_proj = nn.Linear(
            self.encoder.config.hidden_size,
            decoder.token_embedding.embedding_dim
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        tgt,
        tgt_key_padding_mask=None,
    ):
        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)
            memory = encoder_out.last_hidden_state # (B, T, d_enc)

        memory = self.memory_proj(memory) # (B, T, d_dec)

        decoder_logits = self.decoder(
            tgt,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory=memory
        )

        return decoder_logits


    def generate(
        self,
        input_ids,
        attention_mask,
        bos_id,
        eos_id,
        max_len=128,
        decoding_strategy="top_p",
        top_p=0.9,
        device=None,
    ):
        device = device or input_ids.device

        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask)
            memory = encoder_out.last_hidden_state
            memory = self.memory_proj(memory)

        generated_ids = self.decoder.generate(
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            decoding_strategy=decoding_strategy,
            top_p=top_p,
            device=device,
            memory=memory,
        )

        return generated_ids
