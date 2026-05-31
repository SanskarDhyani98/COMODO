"""
Smoke test for COMODO: downloads pretrained models and runs a forward pass
with synthetic IMU data. No real dataset required.
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "comodo")

from comodo.utils.model_util import create_pipeline, IMUStudentMLP
from comodo.loss import COMODOLoss

DEVICE = torch.device("cpu")
BATCH_SIZE = 4
N_CHANNELS = 6
SEQ_LEN = 1000
MLP_HIDDEN = 256
MLP_OUT = 128
QUEUE_SIZE = 16

print("=" * 60)
print("COMODO Smoke Test")
print("=" * 60)

# ── 1. Load pretrained IMU student (Mantis-8M from HuggingFace) ──────────
print("\n[1/4] Loading pretrained Mantis-8M IMU backbone...")
backbone = create_pipeline("paris-noah/Mantis-8M", num_classes=8, device=DEVICE, reduction="concat")
print("     Backbone loaded.")

# ── 2. Wrap with MLP projection head ────────────────────────────────────
print("\n[2/4] Wrapping backbone with IMUStudentMLP projection head...")
imu_student = IMUStudentMLP(
    backbone,
    device=DEVICE,
    mlp_output_dim=MLP_OUT,
    mlp_hidden_dim=MLP_HIDDEN,
    reduction="concat",
)
imu_student.eval()
n_params = sum(p.numel() for p in imu_student.parameters())
print(f"     Total parameters: {n_params:,}")

# ── 3. Synthetic IMU data (batch of 6-channel time series) ───────────────
print("\n[3/4] Running forward pass on synthetic IMU data...")
imu_data = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN)

with torch.no_grad():
    # Mantis needs seq_len=512; resize is handled internally
    embeddings = imu_student(imu_data)

print(f"     Input  shape : {tuple(imu_data.shape)}")
print(f"     Output shape : {tuple(embeddings.shape)}")
print(f"     Embedding dim: {embeddings.shape[-1]}")

# ── 4. COMODO loss with synthetic video embeddings ──────────────────────
print("\n[4/4] Computing COMODOLoss with synthetic video queue...")
# Simulate a pre-encoded video queue (teacher embeddings)
instance_queue = torch.randn(QUEUE_SIZE, MLP_OUT)
video_batch    = torch.randn(BATCH_SIZE, MLP_OUT)

comodo_loss = COMODOLoss(
    instanceQ_encoded=instance_queue,
    student_model=imu_student,
    teacher_temp=0.1,
    student_temp=0.05,
)

loss_val = comodo_loss(imu_data, video_batch)
print(f"     COMODOLoss value: {loss_val.item():.4f}")

print("\n" + "=" * 60)
print("Smoke test PASSED — all components work correctly.")
print("=" * 60)
print("\nTo run full training you need:")
print("  1. Download Ego4D / EgoExo4D / UESTC-MMEA-CL dataset files")
print("  2. Run:  python comodo/train.py --video_ckpt <ckpt> --imu_ckpt <ckpt> --dataset_path <path>")
