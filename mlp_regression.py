import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RegrMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(1)

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    xs = StandardScaler().fit(X_train)
    X_train, X_val = xs.transform(X_train), xs.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)

    tr = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    va = DataLoader(TensorDataset(X_val, y_val), batch_size=512)

    model = RegrMLP(in_dim=X_train.size(1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            mse_sum, n = 0.0, 0
            for xb, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mse = loss_fn(model(xb), yb)
                mse_sum += mse.item() * yb.size(0)
                n += yb.size(0)
        rmse = (mse_sum/n) ** 0.5
        print(f"epoch={epoch:02d} val_RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
