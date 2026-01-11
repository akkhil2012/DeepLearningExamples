import json, os, time
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    seed: int = 123
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 64
    ckpt_dir: str = "checkpoints"

def set_reproducible(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For full determinism you may need env var CUBLAS_WORKSPACE_CONFIG, etc.

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

def make_data(n=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 10, generator=g)
    y = (X[:, 0] + 0.5*X[:, 1] > 0).long()
    return X, y

def log_json(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def save_ckpt(cfg, model, opt, epoch):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    p = os.path.join(cfg.ckpt_dir, f"epoch_{epoch}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, p)
    return p

def main():
    cfg = Config()
    set_reproducible(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    X, y = make_data(seed=cfg.seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True)

    model = SmallNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    run_id = int(time.time())
    log_path = os.path.join(cfg.ckpt_dir, f"run_{run_id}.jsonl")
    log_json(log_path, {"event": "config", **asdict(cfg)})

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            n += yb.size(0)

        ckpt = save_ckpt(cfg, model, opt, epoch)
        metrics = {"event": "epoch_end", "epoch": epoch, "train_loss": loss_sum/n, "ckpt": ckpt}
        log_json(log_path, metrics)
        print(metrics)

if __name__ == "__main__":
    main()
