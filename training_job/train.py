import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
import json
import os
import time
from pathlib import Path
from model import SmallCNN

# ── load config ────────────────────────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── emit metrics ───────────────────────────────────────────────────────────────
def emit_metrics(metrics_file, payload):
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "a") as f:
        f.write(json.dumps(payload) + "\n")

# ── compute gradient norm ──────────────────────────────────────────────────────
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# ── load data ──────────────────────────────────────────────────────────────────
def get_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"]
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"]
    )
    return trainloader, valloader

# ── validation loop ────────────────────────────────────────────────────────────
def validate(model, valloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    model.train()
    return total_loss / len(valloader), correct / total

# ── main training loop ─────────────────────────────────────────────────────────
def train():
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    trainloader, valloader = get_dataloaders(cfg)

    model = SmallCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()

    if cfg["training"]["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            momentum=0.9,
            weight_decay=cfg["training"]["weight_decay"]
        )

    metrics_file = cfg["monitoring"]["metrics_file"]
    emit_every = cfg["monitoring"]["emit_every_n_steps"]
    grad_clip = cfg["training"]["gradient_clip"]

    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            grad_norm = get_grad_norm(model)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % emit_every == 0:
                val_loss, val_acc = validate(model, valloader, criterion, device)
                payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": round(running_loss / emit_every, 6),
                    "val_loss": round(val_loss, 6),
                    "val_acc": round(val_acc, 6),
                    "grad_norm": round(grad_norm, 6),
                    "timestamp": time.time()
                }
                emit_metrics(metrics_file, payload)
                print(f"step {global_step} | train_loss {payload['train_loss']} | val_loss {payload['val_loss']} | val_acc {payload['val_acc']} | grad_norm {payload['grad_norm']}")
                running_loss = 0.0

        # checkpoint
        if (epoch + 1) % cfg["checkpointing"]["save_every"] == 0:
            Path(cfg["checkpointing"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{cfg['checkpointing']['checkpoint_dir']}epoch_{epoch+1}.pt")
            print(f"checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    train()
