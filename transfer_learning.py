import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train = OxfordIIITPet("./data", split="trainval", target_types="category",
                          download=True, transform=tfm)
    test  = OxfordIIITPet("./data", split="test", target_types="category",
                          download=True, transform=tfm)

    tr = DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
    te = DataLoader(test, batch_size=64, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    # replace head
    num_classes = len(train.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def eval_acc():
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x).argmax(1)
                c += (p == y).sum().item()
                t += y.numel()
        return c/t

    # train head
    for epoch in range(2):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("head-only epoch", epoch, "acc", eval_acc())

    # unfreeze some layers for fine-tune
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(2):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print("finetune epoch", epoch, "acc", eval_acc())

if __name__ == "__main__":
    main()
