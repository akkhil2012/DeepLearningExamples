import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

def make_data(n=2000):
    torch.manual_seed(0)
    x0 = torch.randn(n//2, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(n//2, 2) + torch.tensor([ 2.0, 0.0])
    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n//2), torch.ones(n//2)]).long()
    idx = torch.randperm(n)
    return X[idx], y[idx]

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = ce(logits, y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total

def main():
    X, y = make_data()
    train_X, train_y = X[:1600], y[:1600]
    val_X, val_y     = X[1600:], y[1600:]

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_X, val_y), batch_size=256)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(Xb), yb)
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader)
        print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # save
    path = "model.pt"
    torch.save({"state_dict": model.state_dict()}, path)

    # load
    loaded = MLP().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    loaded.load_state_dict(ckpt["state_dict"])
    loaded.eval()

    # inference
    sample = torch.tensor([[0.0, 0.0], [3.0, 0.0], [-3.0, 0.2]], device=DEVICE)
    probs = loaded(sample).softmax(dim=1)
    print("inference probs:\n", probs)

if __name__ == "__main__":
    main()
