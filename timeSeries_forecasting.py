import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_series(n=5000):
    torch.manual_seed(0)
    t = torch.arange(n).float()
    s = torch.sin(0.01*t) + 0.5*torch.sin(0.03*t) + 0.1*torch.randn(n)
    return s

class WindowDS(Dataset):
    def __init__(self, series, lookback=64, horizon=1):
        self.s = series
        self.L = lookback
        self.H = horizon
    def __len__(self):
        return len(self.s) - self.L - self.H
    def __getitem__(self, i):
        x = self.s[i:i+self.L].unsqueeze(-1)                # [L,1]
        y = self.s[i+self.L:i+self.L+self.H]                # [H]
        return x, y

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)            # [B,L,H]
        last = out[:, -1, :]             # last step
        return self.fc(last).squeeze(-1) # [B]

def main():
    series = make_series()
    train_s, test_s = series[:4000], series[4000-64:]  # overlap for windows

    tr = DataLoader(WindowDS(train_s), batch_size=128, shuffle=True)
    te = DataLoader(WindowDS(test_s), batch_size=256)

    model = LSTMForecaster().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        model.train()
        for x,y in tr:
            x = x.to(DEVICE)
            y = y[:,0].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mse_sum, n = 0.0, 0
            for x,y in te:
                x = x.to(DEVICE); y = y[:,0].to(DEVICE)
                pred = model(x)
                mse_sum += loss_fn(pred, y).item() * y.size(0)
                n += y.size(0)
        rmse = (mse_sum/n) ** 0.5
        print("epoch", epoch, "test_RMSE", rmse)

if __name__ == "__main__":
    main()
