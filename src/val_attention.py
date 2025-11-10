#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import torch

from train import MultimodalFusionModule
from data import create_dataloaders

_CAPTURED = {}

def _avg_heads_batch_tq(att):
    """
    Accept attention in shapes like:
      (B, H, Tq, Tk) or (B, Tq, Tk) or (H, Tq, Tk) or (Tq, Tk)
    Returns a single row vector of length Tk (averaged over B, H, and Tq if needed).
    """
    a = att
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu()
    a = torch.as_tensor(a)

    if a.ndim == 4:
        a = a.mean(dim=(0,1,2))
    elif a.ndim == 3:
        a = a.mean(dim=(0,1))   
    elif a.ndim == 2:
        if a.size(0) > 1:
            a = a.mean(dim=0)    
        else:
            a = a.squeeze(0)  
    elif a.ndim == 1:
        pass
    else:
        raise ValueError(f"Unsupported attention shape: {tuple(a.shape)}")
    return a.numpy()

def _make_hook(mod_name):
    """
    Hook for CrossModalAttention layers that return (out, attn).
    Stores attention under the modality that *queried* (layer key).
    """
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            _, attn = outputs[:2]
        else:
            attn = getattr(module, "last_attn", None)
            if attn is None:
                return
        _CAPTURED[mod_name] = attn.detach().cpu()
    return hook

@torch.no_grad()
def main(ckpt_path: str, out_dir: str = "experiments/attn", device: str = "cpu"):
    out_dir = Path(out_dir)
    (out_dir / "per_modality").mkdir(parents=True, exist_ok=True)

    model = MultimodalFusionModule.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    cfg = model.config

    fusion = getattr(model, "fusion_model", None)
    assert fusion is not None, "Model has no fusion_model"

    hooks = []
    modality_names = None

    if hasattr(fusion, "modality_names"):
        modality_names = list(fusion.modality_names)

    if hasattr(fusion, "attn_layers") and isinstance(fusion.attn_layers, torch.nn.ModuleDict):
        for name, layer in fusion.attn_layers.items():
            hooks.append(layer.register_forward_hook(_make_hook(name)))
    elif hasattr(fusion, "attn"):  
        hooks.append(fusion.attn.register_forward_hook(_make_hook("shared")))
    else:
        raise RuntimeError("No attention layers found (expected fusion.attn_layers or fusion.attn).")

    _, val_loader, _ = create_dataloaders(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        modalities=cfg.dataset.modalities,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers
    )

    batch = next(iter(val_loader))
    features, labels, mask = batch
    features = {k: v.to(device) for k, v in features.items()}
    mask = mask.to(device)

    _ = model(features, mask)

    if modality_names is None:
        modality_names = sorted(_CAPTURED.keys())
    M = len(modality_names)
    if M == 0:
        if "shared" in _CAPTURED and hasattr(cfg.dataset, "modalities"):
            modality_names = list(cfg.dataset.modalities)
            M = len(modality_names)
        else:
            raise RuntimeError("No attention captured. Ensure HybridFusion uses CrossModalAttention and that a forward pass happened.")

    name_to_idx = {m: i for i, m in enumerate(modality_names)}
    MxM = np.zeros((M, M), dtype=np.float32)

    have_per_modality = any(k in name_to_idx for k in _CAPTURED.keys())

    if have_per_modality:
        for q_name, attn in _CAPTURED.items():
            if q_name not in name_to_idx:
                continue
            row = _avg_heads_batch_tq(attn)  # -> (Tk,)
            if row.shape[0] != M:
                raise RuntimeError(f"Captured row length {row.shape[0]} != num modalities {M}.")
            i = name_to_idx[q_name]
            MxM[i, :] = row
            np.save(out_dir / "per_modality" / f"{q_name}.npy", attn.detach().cpu().numpy())
    else:
        attn = next(iter(_CAPTURED.values()))
        mat = attn
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().float().numpy()
        while mat.ndim > 2:
            mat = mat.mean(axis=0)
        if mat.shape != (M, M):
            raise RuntimeError(f"Shared attention after averaging is {mat.shape}, expected ({M},{M}).")
        MxM = mat
        np.save(out_dir / "per_modality" / "shared.npy", next(iter(_CAPTURED.values())).detach().cpu().numpy())

    np.save(out_dir / "attn_MxM.npy", MxM)
    with open(out_dir / "modalities.txt", "w") as f:
        f.write("\n".join(modality_names))

    print("Saved:")
    print(" ", out_dir / "attn_MxM.npy")
    print(" ", out_dir / "modalities.txt")
    print(" ", out_dir / "per_modality/*.npy")

    for h in hooks:
        h.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best/best.ckpt or specific .ckpt")
    parser.add_argument("--out_dir", default="experiments/attn")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.ckpt, args.out_dir, args.device)
