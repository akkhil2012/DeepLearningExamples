import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        def C(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c,out_c,3,1,1), nn.ReLU(),
                                                nn.Conv2d(out_c,out_c,3,1,1), nn.ReLU())
        self.enc1 = C(in_ch, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = C(32, 64);    self.pool2 = nn.MaxPool2d(2)
        self.mid  = C(64, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = C(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = C(64, 32)
        self.out  = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        m  = self.mid(p2)
        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

def dice(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps) / (union + eps)

def main():
    img_tfm = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    # segmentation masks: target_types="segmentation" gives 3-class trimap; make binary for pet vs bg
    ds = OxfordIIITPet("./data", split="trainval", target_types="segmentation", download=True,
                      transform=img_tfm, target_transform=transforms.Compose([
                          transforms.Resize((128,128), interpolation=transforms.InterpolationMode.NEAREST),
                          transforms.PILToTensor(),
                      ]))
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

    model = UNetSmall(out_ch=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        model.train()
        dice_sum, n = 0.0, 0
        for x, mask in loader:
            x = x.to(DEVICE)
            # mask is [B,1,H,W] with values {1,2,3} (trimap). Treat {2,3} as pet, {1} as bg
            mask = mask.to(DEVICE).float()
            y = (mask != 1.0).float()  # [B,1,H,W] binary

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            with torch.no_grad():
                dice_sum += float(dice(logits, y))
                n += 1
        print(f"epoch={epoch} loss={loss.item():.4f} dice={dice_sum/n:.4f}")

if __name__ == "__main__":
    main()
