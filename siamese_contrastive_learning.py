import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PairMNIST(Dataset):
    def __init__(self, root="./data", train=True):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transforms.ToTensor())
        self.by_label = {i: [] for i in range(10)}
        for idx, (_, y) in enumerate(self.ds):
            self.by_label[int(y)].append(idx)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        x1, y1 = self.ds[idx]
        if torch.rand(1).item() < 0.5:
            # positive pair
            idx2 = self.by_label[int(y1)][torch.randint(len(self.by_label[int(y1)]), (1,)).item()]
            x2, y2 = self.ds[idx2]
            label = torch.tensor(1.0)
        else:
            # negative pair
            y2 = int((int(y1) + torch.randint(1,10,(1,)).item()) % 10)
            idx2 = self.by_label[y2][torch.randint(len(self.by_label[y2]), (1,)).item()]
            x2, _ = self.ds[idx2]
            label = torch.tensor(0.0)
        return x1, x2, label

class EmbNet(nn.Module):
    def __init__(self, emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, emb),
        )
    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=1)

def contrastive_loss(z1, z2, same, margin=1.0):
    d = (z1 - z2).pow(2).sum(dim=1).sqrt()
    pos = same * d.pow(2)
    neg = (1 - same) * torch.clamp(margin - d, min=0).pow(2)
    return (pos + neg).mean()

def main():
    ds = PairMNIST(train=True)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)

    model = EmbNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        for x1, x2, same in dl:
            x1, x2, same = x1.to(DEVICE), x2.to(DEVICE), same.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            z1, z2 = model(x1), model(x2)
            loss = contrastive_loss(z1, z2, same)
            loss.backward()
            opt.step()
        print("epoch", epoch, "loss", float(loss.detach().cpu()))

if __name__ == "__main__":
    main()
