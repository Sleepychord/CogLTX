import torch
class MemoryBank():
    def __init__(self, *args, device='cpu'):
        self.data = torch.zeros(*args, device=device)
        self.pos = torch.zeros(args[0], dtype=torch.long, device='cpu')
        self.p = 0
    def __len__(self):
        return len(self.data)
    def __getitem__(self, k):
        return self.data[k]
    def get_addr(self, size):
        if self.p + size > self.__len__():
            self.p = 0
            l, r = len(self.data) - size, len(self.data)
        else:
            self.p += size
            l, r = self.p - size, self.p
        return self.data[l:r], self.pos[l:r]
