import copy
import torch, torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

def make_data(n=4096):
    torch.manual_seed(0)
    X = torch.randn(n, 20)
    y = (X[:, 0] - 0.2*X[:, 1] + 0.1*X[:, 2] > 0).long()
    return X.to(DEVICE), y.to(DEVICE)

def run(optimizer_name):
    X, y = make_data()
    model = TinyNet().to(DEVICE)
    base = copy.deepcopy(model.state_dict())

    if optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    elif optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    else:  # AdamW
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    model.load_state_dict(base)
    ce = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(50):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = ce(model(X), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses

def main():
    results = {name: run(name) for name in ["SGD", "Adam", "AdamW"]}
    for name, ls in results.items():
        plt.plot(ls, label=name)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Optimizer + LR schedule convergence")
    plt.show()

if __name__ == "__main__":
    main()
