import torch
import yaml
from omegaconf import OmegaConf
from pathlib import Path

from train import MultimodalFusionModule     
from uncertainty import MCDropoutUncertainty     

ckpt_path = "runs/a2_hybrid_pamap2/best.ckpt"  
config_yaml = "config/base.yaml"                               

with open(config_yaml, "r") as f:
    cfg = OmegaConf.create(yaml.safe_load(f))

model = MultimodalFusionModule(cfg)


ckpt = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

mc = MCDropoutUncertainty(model=model, num_samples=20)

B = 8
features = {
    "imu": torch.randn(B, cfg.model.encoders.imu.input_dim),
    "hr":  torch.randn(B, cfg.model.encoders.hr.input_dim),
}
mask = torch.ones(B, len(cfg.dataset.modalities))  

with torch.no_grad():
    mean_logits, uncertainty = mc(features, mask) 
    preds = mean_logits.argmax(dim=1)

print("mean_logits:", mean_logits.shape)
print("uncertainty:", uncertainty.shape)
print("preds:", preds)
