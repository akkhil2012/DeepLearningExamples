import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def acc(model, loader):
    model.eval()
    c,t=0,0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        c += (p==y).sum().item()
        t += y.numel()
    return c/t

def main():
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    tr = DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
    te = DataLoader(test, batch_size=64, num_workers=2)

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    model.to(DEVICE)

    opt = torch.optim.AdamW(model.heads.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        for x,y in tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("epoch", epoch, "acc", acc(model, te))

if __name__ == "__main__":
    main()
