import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPT_Dataset(Dataset):
    def __init__(self, tokenizer, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        encoded_text = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(encoded_text)-max_length, stride):
            input_chunk = encoded_text[i : i+max_length]
            target_chunk = encoded_text[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader(txt_files, max_length = 256, stride = 128, batch_size= 4,
                       shuffle=True, num_workers=0, drop_last=True):

    tokenizer = tiktoken.encoding_for_model("gpt-2")
    dataset = GPT_Dataset(tokenizer, txt_files, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataloader