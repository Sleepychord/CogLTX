import torch
class MemoryBank():
    def __init__(self, *args, device='cpu'):
        self.data = torch.zeros(*args, device=device)
        self.p = 0
    def __len__(self):
        return len(self.data)
    def __getitem__(self, k):
        return self.data[k]
    def get_addr(self, size):
        if self.p + size > self.__len__():
            self.p = 0
        self.p += size
        return self.data[self.p - size: self.p]
