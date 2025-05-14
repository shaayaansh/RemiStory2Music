import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from miditok import REMI, TokenizerConfig
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from pretrain import load_checkpoint
from model import Story2MusicTransformer
from dataset import StoryMidiDataset
from remi_decoder import RemiDecoder
from torch.utils.data import DataLoader
import logging
import json
import pickle
import os


def main():
    os.makedirs("finetune_checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "bert-base-uncased"

    logging.basicConfig(
        filename='finetune_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    midi_tokenizer = REMI(config)
    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ft_df = pd.read_csv("finetune_data/EMOPIA_train_data.csv")
    midis_path = "finetune_data/EMOPIA_1.0/midis"
    ft_dataset = StoryMidiDataset(
        ft_df,
        midis_path=midis_path,
        midi_tokenizer=midi_tokenizer,
        text_tokenizer=text_tokenizer,
        max_length=256
    )
    train_dataloader = DataLoader(ft_dataset, shuffle=True, batch_size=8)

    # load the pretrained decoder
    decoder = RemiDecoder(
        len(midi_tokenizer.vocab)
    )
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=1e-4)
    decoder, _, _ = load_checkpoint(decoder, decoder_optimizer, "pretrain_checkpoints")
    print("\n decoder checkpoint loaded! \n")
    model = Story2MusicTransformer(MODEL_NAME, decoder)
    model.to(device)
    
    # freeze everything but last decoder layer
    for layer in model.decoder.decoder.layers[:-1]:  # all but the last
        for param in layer.parameters():
            param.requires_grad = False
    # freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # unfreeze token_embeddings so model learns token relationship with emotions
    for param in model.decoder.token_embedding.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    criterion = nn.CrossEntropyLoss(ignore_index=midi_tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    save_every = 5

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids, attn_mask, tgt = [x.to(device) for x in batch]

            tgt_input = tgt[:, :-1]  
            tgt_target = tgt[:, 1:]

            tgt_key_padding_mask = (tgt_input == midi_tokenizer.pad_token_id)

            logits = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                tgt=tgt_input,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_target_flat = tgt_target.reshape(-1)
            loss = criterion(logits_flat, tgt_target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        
        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch} — Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"finetune_checkpoints/story2music_epoch_{epoch+1}.pt")
    


if __name__ == "__main__":
    main()



