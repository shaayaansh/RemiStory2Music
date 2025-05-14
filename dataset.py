from torch.utils.data import Dataset
from transformers import AutoTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
from miditok import CPWord, TokenizerConfig
import torch


class StoryMidiDataset(Dataset):
    def __init__(self, dataframe, midis_path, midi_tokenizer, text_tokenizer, max_length=512):
        self.df = dataframe
        self.midi_tokenizer = midi_tokenizer
        self.text_tokenizer = text_tokenizer
        self.midi_paths = midis_path
        self.midi_max_length = max_length
        self.pad_token = tuple(midi_tokenizer.pad_token_id for _ in range(len(midi_tokenizer.vocab)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokenized = self.text_tokenizer(
            self.df.iloc[idx]['matched_story'], 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=512
        )
        input_ids, attn_mask = tokenized['input_ids'].squeeze(0), tokenized['attention_mask'].squeeze(0)
        midi_id = self.df.iloc[idx]['ID']
        midi_file_path = Path(self.midi_paths, f"{midi_id}.mid")
        midi_tokenized = self.midi_tokenizer(midi_file_path)
        midi_ids = midi_tokenized[0].ids  # List[List[int]] (T, F)
        
        # Pad or truncate
        midi_tensor = torch.tensor(midi_ids, dtype=torch.long)
        T = midi_tensor.size(0)
        
        if T < self.midi_max_length:
            pad_len = self.midi_max_length - T
            pad_tensor = torch.full((pad_len,), self.midi_tokenizer.pad_token_id, dtype=torch.long)
            midi_tensor = torch.cat([midi_tensor, pad_tensor])
        else:
            midi_tensor = midi_tensor[:self.midi_max_length]

        
        return input_ids, attn_mask, midi_tensor  # shapes: (L,), (L,), (T, F)