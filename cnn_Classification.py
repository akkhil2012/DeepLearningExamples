import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    c, t = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        c += (p == y).sum().item()
        t += y.numel()
    return c/t

def main():
    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

    tr = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    te = DataLoader(test, batch_size=256, num_workers=2)

    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("epoch", epoch, "test_acc", eval_acc(model, te))

if __name__ == "__main__":
    main()
