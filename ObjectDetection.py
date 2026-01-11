import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def collate(batch):
    images, targets = [], []
    for img, ann in batch:
        images.append(img)
        # NOTE: VOC annotations need parsing into boxes/labels.
        # This is a minimal skeleton; implement voc_to_target(ann) for real training.
        targets.append({"boxes": torch.zeros((0,4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64)})
    return images, targets

def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = VOCDetection("./data", year="2007", image_set="train", download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 21  # VOC classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    model.train()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        losses = model(images, targets)  # dict of detection losses
        loss = sum(losses.values())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print("step", step, {k: float(v.detach().cpu()) for k, v in losses.items()})
        if step == 30:
            break

if __name__ == "__main__":
    main()
