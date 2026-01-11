"""
Minimal *skeleton* to show architecture + training loop.
For real captioning: use Flickr8k loader, proper tokenizer, padding masks, and teacher forcing.
"""
import torch, torch.nn as nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EncoderCNN(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # [B,512,H/32,W/32]
        self.proj = nn.Conv2d(512, d, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)              # [B,d,h,w]
        B,d,h,w = feat.shape
        return feat.flatten(2).transpose(1,2)  # [B, hw, d] tokens

class Decoder(nn.Module):
    def __init__(self, vocab, d=256, nhead=4, nlayers=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d, vocab)

    def forward(self, memory, captions_in):
        # captions_in: [B,T]
        B,T = captions_in.shape
        tgt = self.tok(captions_in)  # [B,T,d]
        # causal mask
        mask = torch.triu(torch.ones(T,T, device=tgt.device), diagonal=1).bool()
        h = self.dec(tgt=tgt, memory=memory, tgt_mask=mask)
        return self.out(h)  # [B,T,vocab]

def main():
    vocab = 1000
    enc = EncoderCNN().to(DEVICE)
    dec = Decoder(vocab=vocab).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # dummy batch (replace with Flickr8k)
    images = torch.randn(8,3,224,224, device=DEVICE)
    captions_in  = torch.randint(1, vocab, (8, 16), device=DEVICE)
    captions_out = torch.randint(1, vocab, (8, 16), device=DEVICE)

    for step in range(10):
        opt.zero_grad(set_to_none=True)
        memory = enc(images)
        logits = dec(memory, captions_in)
        loss = loss_fn(logits.reshape(-1, vocab), captions_out.reshape(-1))
        loss.backward()
        opt.step()
        print("step", step, "loss", float(loss))

if __name__ == "__main__":
    main()
