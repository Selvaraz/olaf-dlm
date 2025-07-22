import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class OpalDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = []
        self.target_ids = []
        self._prepare_data(txt)


    def _prepare_data(self, txt):
        token_ids = self.tokenizer.encode(txt)
        print(f"Preparing the tensors for length: {len(token_ids) - self.max_length}")
        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1:i + self.max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        # return {
        #     'input_ids': self.input_ids[idx],
        #     'target_ids': self.target_ids[idx]
        # }
    

# def execute_test(createOpalDataLoader):
#     txt = None
#     with open("data/the-verdict.txt", "r") as f:
#         txt = f.read()
# #887,   673,  3521,   470
#     dataloader = createOpalDataLoader(txt, batch_size=5, max_length=4, stride=3, 
#                                     shuffle=False, drop_last=True, num_workers=0)

#     dataIter = iter(dataloader)
#     first_batch = next(dataIter)
#     second_batch = next(dataIter)

#     print(f"Input IDs: ")
#     print(first_batch['input_ids'])

#     print(f"Target IDs: ")
#     print(first_batch['target_ids'])

#     print(" ---- Second Batch ---- ")
#     print(f"Input IDs: ")
#     print(second_batch['input_ids'])

#     print(f"Target IDs: ")
#     print(second_batch['target_ids'])

# #execute_test(createOpalDataLoader)
