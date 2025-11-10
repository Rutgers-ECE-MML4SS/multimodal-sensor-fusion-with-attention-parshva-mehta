import json
from pathlib import Path
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import time

from train import MultimodalFusionModule
from data import create_dataloaders
from uncertainty import CalibrationMetrics


def _synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def evaluate_ckpt(ckpt_path: Path, device="cpu"):
    model = MultimodalFusionModule.load_from_checkpoint(str(ckpt_path))
    model.eval().to(device)

    cfg = model.config
    _, _, test_loader = create_dataloaders(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        modalities=cfg.dataset.modalities,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers
    )

    all_preds, all_labels, all_confs = [], [], []
    total_loss, n_batches = 0, 0

    # ---- Inference timing ----
    # warm up a few batches so timing is stable (esp. on CUDA)
    warmup = 3
    for i, (features, labels, mask) in enumerate(test_loader):
        if i >= warmup:
            break
        with torch.no_grad():
            features = {k: v.to(device) for k, v in features.items()}
            _ = model(features, mask.to(device))
    # real timing
    total_forward_time = 0.0
    total_examples_timed = 0

    with torch.no_grad():
        for features, labels, mask in tqdm(test_loader, desc=f"Testing {cfg.model.fusion_type}"):
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            mask = mask.to(device)

            # time forward pass
            _synchronize(device)
            t0 = time.perf_counter()
            logits = model(features, mask)
            _synchronize(device)
            t1 = time.perf_counter()

            total_forward_time += (t1 - t0)
            total_examples_timed += labels.size(0)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_confs.append(confs.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    confs = torch.cat(all_confs)

    accuracy = (preds == labels).float().mean().item()
    f1_macro = f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    ece = CalibrationMetrics.expected_calibration_error(confs, preds, labels)
    avg_loss = total_loss / max(1, n_batches)

    # ms per sample
    inference_ms = (total_forward_time / max(1, total_examples_timed)) * 1000.0

    return {
        "fusion_type": cfg.model.fusion_type,
        "dataset": cfg.dataset.name,
        "modalities": list(cfg.dataset.modalities),
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "ece": ece,
            "loss": avg_loss,
            "inference_ms": inference_ms,
        },
        "ckpt_path": str(ckpt_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare fusion checkpoints")
    parser.add_argument("--early", required=True, help="Path to early fusion .ckpt")
    parser.add_argument("--hybrid", required=True, help="Path to hybrid fusion .ckpt")
    parser.add_argument("--late", required=True, help="Path to late fusion .ckpt")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--output", default="experiments/fusion_comparison.json")
    args = parser.parse_args()

    ckpt_paths = {
        "early_fusion": Path(args.early),
        "late_fusion": Path(args.late),
        "hybrid_fusion": Path(args.hybrid)
    }

    results = {}
    dataset, modalities = None, None

    for name, ckpt in ckpt_paths.items():
        out = evaluate_ckpt(ckpt, device=args.device)
        results[name] = out["metrics"]
        dataset = dataset or out["dataset"]
        modalities = modalities or out["modalities"]

    report = {
        "dataset": dataset,
        "modalities": modalities,
        "results": results
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFusion comparison saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
