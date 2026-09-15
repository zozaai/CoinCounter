"""Train the regression engine: python -m benchmark.train_regression --dataset dataset --out models/regression.

Fine-tunes an ImageNet ResNet-18 to predict the coin count with a sigmoid-bounded output in
[0, max_count]. Trains on the train split, selects the best epoch on val by exact-count
accuracy then MAE, and never reads the test split.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coincounter.engines.regression import IMAGENET_MEAN, IMAGENET_STD, bounded_count, build_model
from .dataset import samples


def make_dataset(torch, transforms, folder, size, train):
    from PIL import Image

    class Split(torch.utils.data.Dataset):
        def __init__(self):
            self.items = samples(folder.parent, folder.name)
            augment = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                       transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)] if train else []
            self.transform = transforms.Compose([transforms.Resize((size, size))] + augment +
                                                [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, count = self.items[index]
            with Image.open(path) as source:
                tensor = self.transform(source.convert("RGB"))
            if train:
                tensor = torch.rot90(tensor, random.randint(0, 3), dims=(1, 2))
            return tensor, torch.tensor(float(count))
    return Split()


def evaluate(torch, model, loader, device, max_count):
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions += bounded_count(model(x.to(device)), max_count).cpu().tolist()
            actual += y.tolist()
    rounded, actual = np.rint(predictions), np.array(actual)
    return {"accuracy": float(np.mean(rounded == actual)), "mae": float(np.mean(np.abs(rounded - actual))),
            "raw_mae": float(np.mean(np.abs(np.array(predictions) - actual)))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "models/regression")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--max-count", type=float, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    import torch
    from torchvision import transforms
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    train = torch.utils.data.DataLoader(make_dataset(torch, transforms, args.dataset / "train", args.image_size, True),
                                        args.batch, shuffle=True)
    val = torch.utils.data.DataLoader(make_dataset(torch, transforms, args.dataset / "val", args.image_size, False), args.batch)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train))
    loss_fn = torch.nn.SmoothL1Loss()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.out / f"resnet18-{args.image_size}.pt"
    best, history, started = None, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train:
            optimizer.zero_grad()
            loss = loss_fn(bounded_count(model(x.to(device)), args.max_count), y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
        metrics = evaluate(torch, model, val, device, args.max_count)
        metrics.update(epoch=epoch, train_loss=total / len(train.dataset))
        history.append(metrics)
        print(f"epoch {epoch:02d} loss {metrics['train_loss']:.3f} val accuracy {metrics['accuracy']:.3f} "
              f"mae {metrics['mae']:.3f}", file=sys.stderr, flush=True)
        key = (metrics["accuracy"], -metrics["mae"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, **metrics}
            torch.save(model.state_dict(), weights)
    best.pop("key")
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "architecture": "torchvision resnet18 (IMAGENET1K_V1 init), fc -> 1, count = max_count * sigmoid",
              "selection": "best val exact-count accuracy, then MAE; test split never read",
              "best": best, "history": history, "weights": weights.name,
              "seconds": time.time() - started, "torch": torch.__version__}
    (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"best epoch {best['epoch']}: val accuracy {best['accuracy']:.3f}, MAE {best['mae']:.3f}; "
          f"saved {weights} in {report['seconds']:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
