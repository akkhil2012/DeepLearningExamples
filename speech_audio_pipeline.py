import torch, torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "./data"
TARGET_SR = 16000
TARGET_SAMPLES = TARGET_SR  # 1 sec
mel = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR, n_fft=1024, hop_length=256, n_mels=64)
db  = torchaudio.transforms.AmplitudeToDB(stype="power")

class Subset(SPEECHCOMMANDS):
    def __init__(self, root, subset):
        super().__init__(root, download=True)
        def load_list(fn):
            with open(Path(self._path)/fn) as f:
                return set(str(Path(self._path)/line.strip()) for line in f)
        valid = load_list("validation_list.txt")
        test  = load_list("testing_list.txt")
        if subset == "validation":
            self._walker = [w for w in self._walker if w in valid]
        elif subset == "testing":
            self._walker = [w for w in self._walker if w in test]
        else:
            self._walker = [w for w in self._walker if (w not in valid and w not in test)]

def label_of(path): return Path(path).parts[-2]

train_ds = Subset(ROOT, "training")
labels = sorted({label_of(p) for p in train_ds._walker})
lab2id = {l:i for i,l in enumerate(labels)}

def fix_len(w):
    if w.size(1) > TARGET_SAMPLES: return w[:, :TARGET_SAMPLES]
    if w.size(1) < TARGET_SAMPLES: return torch.nn.functional.pad(w, (0, TARGET_SAMPLES - w.size(1)))
    return w

def preprocess(w, sr):
    if w.size(0) > 1: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR: w = torchaudio.functional.resample(w, sr, TARGET_SR)
    w = fix_len(w)
    with torch.no_grad():
        x = db(mel(w))  # [1, 64, T]
    return x

class Wrapped(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        w, sr, *_ = self.base[idx]
        path = self.base._walker[idx]
        x = preprocess(w, sr)
        y = lab2id[label_of(path)]
        return x, y

def collate(batch):
    xs = torch.stack([b[0] for b in batch])  # [B,1,64,T]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys

class AudioCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(32, n)
    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

def main():
    train = DataLoader(Wrapped(train_ds), batch_size=64, shuffle=True, num_workers=2, collate_fn=collate)
    model = AudioCNN(len(labels)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        for x,y in train:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("epoch", epoch, "loss", float(loss.detach().cpu()))

if __name__ == "__main__":
    main()
