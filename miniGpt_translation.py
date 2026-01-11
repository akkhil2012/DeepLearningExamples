import math, torch, torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = "hello world\n" * 5000
chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(batch_size=64, block_size=64):
    ix = torch.randint(0, len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, nhead=4, nlayers=2, block=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm = nn.Linear(d, vocab)
        self.block = block

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)
        return self.lm(h)

@torch.no_grad()
def sample(model, start="h", steps=50):
    model.eval()
    x = torch.tensor([[stoi[start]]], device=DEVICE)
    for _ in range(steps):
        logits = model(x[:, -model.block:])
        probs = logits[:, -1].softmax(-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    return "".join(itos[i] for i in x[0].tolist())

def main():
    model = TinyGPT(vocab=len(chars)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(500):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print("step", step, "loss", float(loss), "sample:", sample(model))

if __name__ == "__main__":
    main()
