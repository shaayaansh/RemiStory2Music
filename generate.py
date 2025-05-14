from miditok import REMI, TokenizerConfig
from remi_decoder import RemiDecoder
import torch
import torch.nn as nn
import pickle
from miditok import TokSequence
import os
from tqdm import tqdm
from utils import convert_to_midi_files

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

bos_id = 1
eos_id = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RemiDecoder(len(tokenizer.vocab)).to(device)

epoch_to_load = 250
checkpoint = torch.load(f"pretrain_checkpoints/remidecoder_epoch_{epoch_to_load}.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded from checkpoint!")

os.makedirs(f"generated_midis_{epoch_to_load}", exist_ok=True)

print("START GENERATING..")
for i in tqdm(range(30)):
    top_p_ids = model.generate(
        bos_id=bos_id,
        eos_id=eos_id,
        decoding_strategy="top_p",
        top_p=0.9,
        max_len=512,
        device=device
    )

    convert_to_midi_files(
        top_p_ids,
        tokenizer,
        i+1,
        epoch_to_load
    )
