import torch.optim as optim
import torch
import torch.nn as nn
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from utils import load_pretrain_data, split_pretrain_data
from random import shuffle
from miditok.data_augmentation import augment_dataset
from remi_decoder import RemiDecoder
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
import re
import pickle
import torch.nn.functional as F


def main():
    
    logging.basicConfig(
        filename='remidecoder_pretrain_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs("pretrain_checkpoints", exist_ok=True)

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
    tokenizer = REMI(config)

    # download and split pretrain data only if the folder does not exist
    if not os.path.exists("midis"):
        pretrain_file_id = "1BDEPaEWFEB2ADquS1VYp5iLZYVngw799"
        url = f"https://drive.google.com/uc?id={pretrain_file_id}"
        
        load_pretrain_data(url, "midis.zip", "midis")
        midis_path = list(Path("midis/midis").resolve().glob("**/*.mid"))   

        split_pretrain_data("midis", tokenizer, 256)

    midi_paths = list(Path("pretrain_data/dataset_train").resolve().glob("**/*.mid"))
    val_midi_paths = list(Path("pretrain_data/dataset_validation").resolve().glob("**/*.mid"))

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=128,
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer["EOS_None"],
    )
    
    val_dataset = DatasetMIDI(
        files_paths=val_midi_paths,
        tokenizer=tokenizer,
        max_seq_len=128,
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer["EOS_None"],
    )
    
    batch_size = 256
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collator, batch_size=batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RemiDecoder(
        len(tokenizer.vocab)
    )

    # use multiple gpu if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    print(f"model moved to {device}!")
    
    warmup_steps = 1000
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # load from checkpoints
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, "pretrain_checkpoints")
    print(f"Loaded checkpoints! starting training from EPOCH: {start_epoch}: ")

    num_epochs = 400
    save_every = 5
    val_every = 2
    log_interval = 1000

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, batch in enumerate(tqdm(data_loader)):
            input_ids = batch['input_ids'].to(device) # (B, T)
            attention_mask = batch['attention_mask'].to(device)
            
            decoder_input = input_ids[:,:-1] # (B, T-1)
            attn_mask = attention_mask[:, :-1] 
            tgt = input_ids[:,1:] # (B, T-1)
            tgt_key_padding_mask = (attn_mask == 0)

            logits = model(
                decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory=None
            ) # (B, T-1, vocab_size)
            
            logits_flat = logits.reshape(-1, logits.size(-1))  # (B * T-1, vocab_size)
            tgt_flat = tgt.reshape(-1)                         # (B * T-1)

            loss = criterion(logits_flat, tgt_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            if step % log_interval == 0:
                log_msg = f"step {step} - Loss: {loss:.4f}"
                logging.info(log_msg)

            
        log_msg = f"Epoch {epoch} — Loss: {epoch_loss / len(data_loader)*batch_size:.4f}"
        print(log_msg)
        logging.info(log_msg)

        if epoch % val_every == 0:
            validate(model, val_dataloader, criterion, device, epoch)
        
        if epoch % save_every == 0 and epoch != 0:
            checkpoint_path = f"pretrain_checkpoints/remidecoder_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

def validate(model, val_dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)       # (B, T)
            attention_mask = batch['attention_mask'].to(device)
            
            decoder_input = input_ids[:, :-1]               # (B, T-1)
            tgt = input_ids[:, 1:]                          # (B, T-1)
            attn_mask = attention_mask[:, :-1]
            tgt_key_padding_mask = (attn_mask == 0)

            logits = model(
                decoder_input,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory=None
            ) # (B, T-1, vocab_size)

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt.reshape(-1)

            valid_tokens = (tgt_flat != criterion.ignore_index).sum().item()
            loss = criterion(logits_flat, tgt_flat)

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    log_msg = f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}"
    print(log_msg)
    logging.info(log_msg)
    model.train()


def load_checkpoint(model, optimizer, checkpoint_dir="pretrain_checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_epoch(filename):
        match = re.search(r"epoch_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    ]
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)

    if not checkpoint_files:
        print("No checkpoints found!")
        return model, optimizer, 0

    latest_ckpt = os.path.join(checkpoint_dir, checkpoint_files[-1])
    print(f"Loading checkpoint: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location=device)

    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    start_epoch = checkpoint["epoch"] + 1

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, start_epoch


if __name__ == "__main__":
    main()
