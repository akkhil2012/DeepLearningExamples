import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    c, t = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        c += (p == y).sum().item()
        t += y.numel()
    return c/t

def main():
    tfm = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    test  = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

    tr_loader = DataLoader(train, batch_size=128, shuffle=True)
    te_loader = DataLoader(test, batch_size=256)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("epoch", epoch, "test_acc", accuracy(model, te_loader))

if __name__ == "__main__":
    main()
