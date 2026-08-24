"""Train the tiny listwise no-ReID association scorer from exported candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from association_model import (
    ASSOCIATION_MODEL_VERSION,
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    AssociationMLP,
)


class CandidateGroups(Dataset):
    """Each item is one track/frame/phase candidate set with one positive."""

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as data:
            self.features = np.asarray(data["features"], dtype=np.float32)
            self.labels = np.asarray(data["labels"], dtype=np.uint8)
            groups = np.asarray(data["groups"], dtype=np.int64)
            names = tuple(str(name) for name in data["feature_names"])
        if self.features.ndim != 2 or self.features.shape[1] != PAIR_FEATURE_DIM:
            raise ValueError(f"{npz_path}: expected [N,{PAIR_FEATURE_DIM}] features")
        if names != PAIR_FEATURE_NAMES:
            raise ValueError(f"{npz_path}: incompatible association feature ordering")
        unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
        self.group_indices = [np.arange(start, start + count) for start, count in zip(starts, counts)]
        self.group_indices = [indices for indices in self.group_indices if self.labels[indices].sum() == 1]
        if not self.group_indices:
            raise ValueError(f"{npz_path}: no groups with exactly one positive")

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, index):
        indices = self.group_indices[index]
        labels = self.labels[indices]
        return self.features[indices], int(np.flatnonzero(labels)[0])


def collate(groups):
    features, positive_indices, offsets = [], [], [0]
    for group_features, positive_index in groups:
        features.append(group_features)
        positive_indices.append(offsets[-1] + positive_index)
        offsets.append(offsets[-1] + len(group_features))
    return (
        torch.from_numpy(np.concatenate(features, axis=0)),
        torch.tensor(positive_indices, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


def listwise_loss_and_accuracy(logits, positive_indices, offsets):
    losses, correct = [], 0
    for group_index, positive_index in enumerate(positive_indices):
        start, end = offsets[group_index].item(), offsets[group_index + 1].item()
        group_logits = logits[start:end]
        losses.append(torch.logsumexp(group_logits, dim=0) - logits[positive_index])
        correct += int(group_logits.argmax().item() == positive_index.item() - start)
    return torch.stack(losses).mean(), correct, len(positive_indices)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_correct = total_groups = 0
    for features, positive, offsets in loader:
        logits = model(features.to(device, non_blocking=True))
        loss, correct, groups = listwise_loss_and_accuracy(logits, positive, offsets)
        total_loss += float(loss) * groups
        total_correct += correct
        total_groups += groups
    return total_loss / total_groups, total_correct / total_groups


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_set, val_set = CandidateGroups(args.train_data), CandidateGroups(args.val_data)
    feature_mean = train_set.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(train_set.features.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4)
    train_set.features = (train_set.features - feature_mean) / feature_std
    val_set.features = (val_set.features - feature_mean) / feature_std
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AssociationMLP(PAIR_FEATURE_DIM, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_set, batch_size=args.batch_groups, shuffle=True, num_workers=0,
                              pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_groups, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda", collate_fn=collate)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_loss, remaining_patience, history = float("inf"), args.patience, []
    print(f"device: {device} | train groups: {len(train_set):,} | val groups: {len(val_set):,}")
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = correct = groups_sum = 0
        for features, positive, offsets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device, non_blocking=True))
            loss, batch_correct, batch_groups = listwise_loss_and_accuracy(logits, positive, offsets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * batch_groups
            correct += batch_correct; groups_sum += batch_groups
        train_loss, train_acc = loss_sum / groups_sum, correct / groups_sum
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_top1": train_acc,
               "val_loss": val_loss, "val_top1": val_acc}
        history.append(row)
        print("epoch {epoch:02d} | train loss {train_loss:.5f} top1 {train_top1:.4f} | "
              "val loss {val_loss:.5f} top1 {val_top1:.4f}".format(**row))
        if val_loss < best_loss - args.min_delta:
            best_loss, remaining_patience = val_loss, args.patience
            torch.save({
                "association_model_version": ASSOCIATION_MODEL_VERSION,
                "model_state_dict": model.state_dict(), "feature_dim": PAIR_FEATURE_DIM,
                "feature_names": PAIR_FEATURE_NAMES, "hidden_dim": args.hidden_dim,
                "feature_mean": feature_mean, "feature_std": feature_std,
                "best_val_loss": best_loss, "args": vars(args),
            }, save_dir / "best_model.pth")
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print("early stopping")
                break
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"best checkpoint: {save_dir / 'best_model.pth'} | val loss {best_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_groups", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
