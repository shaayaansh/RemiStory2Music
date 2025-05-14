import os
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from miditok import REMI, TokenizerConfig
from model import Story2MusicTransformer  
from remi_decoder import RemiDecoder
from tqdm import tqdm
from utils import convert_to_midi_files

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "bert-base-uncased"

    logging.basicConfig(
        filename='generation_log.log',
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Tokenizer setup
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
        "num_tempos": 32,
        "tempo_range": (40, 250),
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    midi_tokenizer = REMI(config)
    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    decoder = RemiDecoder(len(midi_tokenizer.vocab))
    model = Story2MusicTransformer(MODEL_NAME, decoder)
    model.to(device)

    epoch_to_load = 20
    # Load the fine-tuned checkpoint
    checkpoint_path = f"finetune_checkpoints/story2music_epoch_{epoch_to_load}.pt"  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logging.info(f"Loaded fine-tuned model from checkpoint: Epoch {epoch_to_load}")
    print(f"Loaded fine-tuned model from checkpoint: Epoch {epoch_to_load}")

    QUADRANTS = [1, 2, 3, 4]
    bos_id = midi_tokenizer.vocab["BOS_None"]
    eos_id = midi_tokenizer.vocab["EOS_None"]
    for quadrant in QUADRANTS:
        test_data = pd.read_csv(f"finetune_data/test_data/q{quadrant}_test.csv")
        

        for idx, row in test_data.iterrows():
            story_text = row["Prompt"]
            inputs = text_tokenizer(story_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=512,
                decoding_strategy="top_p",
                top_p=0.9,
                device=device
            )

            os.makedirs(f"generated_midis_finetuned_Q{quadrant}_EPOCH{epoch_to_load}", exist_ok=True)
            convert_to_midi_files(
                generated_ids,
                midi_tokenizer,
                idx,
                f"finetuned_Q{quadrant}_EPOCH{epoch_to_load}"
            )

if __name__ == "__main__":
    main()
