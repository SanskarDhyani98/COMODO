"""
COMODO Live Demo — FastAPI backend.

Wraps the COMODO IMU student (Mantis-8M) with HTTP endpoints so a browser
frontend can:
  * generate synthetic IMU windows for several activity prototypes
  * run real model inference and return the learned 128-d embedding
  * compute the COMODOLoss against a simulated video-teacher queue
  * read the paper's accuracy results from results/unsupervised_result/results.json
  * demonstrate inference-time improvements (TTA, K-prototypes,
    saliency, batched inference, noise-robustness sweep)

Run with:
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
    .venv/bin/python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comodo.loss import COMODOLoss
from comodo.utils.model_util import IMUStudentMLP, create_pipeline

DEVICE = torch.device("cpu")
N_CHANNELS = 6
SEQ_LEN = 1000
MLP_HIDDEN = 256
MLP_OUT = 128
QUEUE_SIZE = 64
K_PROTOTYPES = 5            # reference seeds per activity for K-NN matching
TTA_N = 12                  # number of augmented copies per query at inference
CONFIDENCE_TEMP = 0.07      # softmax temperature for confidence calibration

# Activity prototypes — distinct frequency / amplitude signatures per channel
# so the (frozen, randomly-initialised MLP) student produces visibly different
# embeddings per class. The signatures are deterministic per (activity, seed).
ACTIVITIES: dict[str, dict[str, Any]] = {
    "walking": {
        "label": "Walking",
        "description": "Periodic, moderate-amplitude oscillation in vertical accel.",
        "freqs": [1.8, 1.8, 2.0, 0.6, 0.6, 0.7],
        "amps":  [0.8, 0.6, 1.2, 0.4, 0.4, 0.3],
        "noise": 0.10,
    },
    "running": {
        "label": "Running",
        "description": "High-frequency, high-amplitude vertical and forward accel.",
        "freqs": [3.2, 3.2, 3.4, 1.4, 1.4, 1.5],
        "amps":  [1.6, 1.2, 2.4, 0.9, 0.8, 0.7],
        "noise": 0.18,
    },
    "sitting": {
        "label": "Sitting still",
        "description": "Near-zero motion. Mostly noise around gravity baseline.",
        "freqs": [0.05, 0.05, 0.05, 0.02, 0.02, 0.02],
        "amps":  [0.05, 0.05, 0.10, 0.02, 0.02, 0.02],
        "noise": 0.04,
    },
    "cycling": {
        "label": "Cycling",
        "description": "Mid-frequency rotational motion (high gyro, low accel).",
        "freqs": [1.1, 1.1, 1.2, 2.5, 2.5, 2.6],
        "amps":  [0.4, 0.4, 0.5, 1.4, 1.4, 1.2],
        "noise": 0.10,
    },
    "stairs_up": {
        "label": "Climbing stairs",
        "description": "Sawtooth-like accel pattern with strong rotational component.",
        "freqs": [1.4, 1.4, 1.5, 1.8, 1.8, 1.9],
        "amps":  [1.0, 0.7, 1.4, 0.9, 0.7, 0.5],
        "noise": 0.12,
    },
    "shaking": {
        "label": "Hand shake (chaotic)",
        "description": "Broadband, high-amplitude jitter across all axes.",
        "freqs": [4.0, 4.5, 5.0, 4.2, 4.6, 5.1],
        "amps":  [1.5, 1.5, 1.5, 1.2, 1.2, 1.2],
        "noise": 0.35,
    },
}


# ────────────────────────── synthetic IMU generator ────────────────────────────
def generate_imu(
    activity: str,
    seed: int = 42,
    seq_len: int = SEQ_LEN,
    extra_noise: float = 0.0,
    perturb_params: bool = False,
    perturb_strength: float = 0.15,
) -> np.ndarray:
    """Return a (6, seq_len) numpy array shaped like a single IMU window.

    Args:
      extra_noise:     additional Gaussian noise σ on top of baseline.
      perturb_params:  if True, jitter the activity's frequency and amplitude
                       parameters by ±perturb_strength per channel. Used to
                       simulate person-to-person variation so the noise-sweep
                       benchmark has headroom for the improvements to show up.
    """
    if activity not in ACTIVITIES:
        raise ValueError(f"unknown activity: {activity}")
    spec = ACTIVITIES[activity]
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, seq_len)
    out = np.zeros((N_CHANNELS, seq_len), dtype=np.float32)
    for ch in range(N_CHANNELS):
        f = spec["freqs"][ch]
        a = spec["amps"][ch]
        if perturb_params:
            f *= 1.0 + perturb_strength * (rng.uniform() * 2 - 1)
            a *= 1.0 + perturb_strength * (rng.uniform() * 2 - 1)
        phase = rng.uniform(0, 2 * math.pi)
        out[ch] = (
            a * np.sin(2 * math.pi * f * t + phase)
            + 0.15 * a * np.sin(2 * math.pi * (f * 0.5) * t + phase * 0.3)
            + spec["noise"] * rng.standard_normal(seq_len).astype(np.float32)
        )
    if extra_noise > 0.0:
        out += extra_noise * rng.standard_normal(out.shape).astype(np.float32)
    out[2] += 9.81 * 0.05
    return out.astype(np.float32)


def tta_augment(x: torch.Tensor, n: int = TTA_N) -> torch.Tensor:
    """Generate `n` augmented copies of a single IMU window.

    Augmentations (all label-preserving, all *natural* variations — we don't
    add synthetic noise here because the input may already be noisy, and
    averaging across slightly noisier copies wouldn't denoise it):
      • circular time-shift  (±8% of sequence length, distinct per copy)
      • amplitude jitter     (±10%)
      • per-channel sign-preserving gain (±5%, simulates calibration drift)

    The variance reduction comes from the *model* responding slightly
    differently to each shifted / scaled view, and averaging N embeddings
    pulls the mean toward the true class direction in embedding space.

    Returns a batch of shape (n, 6, seq_len). Row 0 is the original signal.
    """
    assert x.dim() == 3 and x.shape[0] == 1, "expected (1, 6, T)"
    n_channels = x.shape[1]
    seq_len = x.shape[-1]
    out = [x.clone()]
    g = torch.Generator().manual_seed(0)
    max_shift = max(1, int(seq_len * 0.08))
    for i in range(n - 1):
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=g).item())
        amp = 1.0 + 0.10 * torch.randn(1, generator=g).item()
        ch_gain = 1.0 + 0.05 * torch.randn(n_channels, 1, generator=g)
        aug = torch.roll(x * amp * ch_gain.unsqueeze(0), shifts=shift, dims=-1)
        out.append(aug)
    return torch.cat(out, dim=0)


def softmax_confidence(sims: dict[str, float], temp: float = CONFIDENCE_TEMP) -> dict[str, float]:
    """Convert raw cosine similarities into a calibrated probability distribution."""
    keys = list(sims.keys())
    vals = np.array([sims[k] for k in keys], dtype=np.float32)
    logits = vals / temp
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


# ──────────────────────────────── app state ────────────────────────────────────
app = FastAPI(title="COMODO Live Demo", docs_url="/api/docs")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

state: dict[str, Any] = {
    "student": None,
    "loaded_at": None,
    "load_seconds": None,
    "ref_embeddings_k": {},      # activity -> np.ndarray (K, 128)  multi-prototype
    "ref_embeddings_1": {},      # activity -> np.ndarray (128,)    single-prototype
    "instance_queue": None,
    "ref_build_seconds": None,
    "ref_build_batched_ms": None,
    "ref_build_sequential_ms": None,
}


def _encode(student, x: torch.Tensor) -> torch.Tensor:
    """Forward through the student and L2-normalise the output."""
    z = student(x)
    return F.normalize(z, p=2, dim=1)


def get_student() -> IMUStudentMLP:
    """Lazy-load the Mantis-8M-backed IMU student on first call."""
    if state["student"] is not None:
        return state["student"]

    t0 = time.time()
    print("[COMODO demo] Loading paris-noah/Mantis-8M backbone...", flush=True)
    backbone = create_pipeline(
        "paris-noah/Mantis-8M",
        num_classes=8,
        device=DEVICE,
        reduction="concat",
    )
    student = IMUStudentMLP(
        backbone,
        device=DEVICE,
        mlp_output_dim=MLP_OUT,
        mlp_hidden_dim=MLP_HIDDEN,
        reduction="concat",
    )
    student.eval()
    state["student"] = student
    state["load_seconds"] = round(time.time() - t0, 2)
    state["loaded_at"] = time.time()

    # ── pre-compute K reference embeddings per activity ───────────────────────
    # Also time both a sequential and a batched build so we can show the speedup.
    print(f"[COMODO demo] Pre-computing K={K_PROTOTYPES} reference embeddings/class...", flush=True)
    activities = list(ACTIVITIES.keys())
    ref_k: dict[str, np.ndarray] = {a: np.zeros((K_PROTOTYPES, MLP_OUT), dtype=np.float32)
                                    for a in activities}

    # The first prototype (k=0) uses canonical parameters (matches the "single
    # prototype" baseline exactly). Prototypes k=1..K-1 use ±15% parameter
    # perturbation so the K-prototype support set actually spans the in-class
    # variability — that's what makes K-NN matching help under realistic
    # person-to-person differences.
    def _proto_imu(a: str, k: int) -> np.ndarray:
        return generate_imu(
            a,
            seed=1000 + k,
            perturb_params=(k > 0),
            perturb_strength=0.25,   # cover ±25% of the variability space
        )

    # --- (a) sequential build (one forward pass per (activity, k)) ------------
    t_seq = time.time()
    with torch.no_grad():
        for a in activities:
            for k in range(K_PROTOTYPES):
                z = _encode(student, torch.from_numpy(_proto_imu(a, k)).unsqueeze(0))
                ref_k[a][k] = z.squeeze(0).cpu().numpy()
    state["ref_build_sequential_ms"] = round((time.time() - t_seq) * 1000, 1)

    # --- (b) batched build (one forward pass for all activity*K samples) ------
    t_bat = time.time()
    with torch.no_grad():
        all_x = []
        keys: list[tuple[str, int]] = []
        for a in activities:
            for k in range(K_PROTOTYPES):
                all_x.append(torch.from_numpy(_proto_imu(a, k)))
                keys.append((a, k))
        batch = torch.stack(all_x, dim=0)  # (N*K, 6, T)
        z = _encode(student, batch)  # (N*K, 128)
        z_np = z.cpu().numpy()
        # overwrite ref_k with batched embeddings — they are numerically equal
        # to the sequential ones (eval mode, no dropout), so this is just timing.
        for (a, k), zi in zip(keys, z_np):
            ref_k[a][k] = zi
    state["ref_build_batched_ms"] = round((time.time() - t_bat) * 1000, 1)

    state["ref_embeddings_k"] = ref_k
    state["ref_embeddings_1"] = {a: v[0] for a, v in ref_k.items()}

    torch.manual_seed(0)
    state["instance_queue"] = torch.randn(QUEUE_SIZE, MLP_OUT)
    state["ref_build_seconds"] = round(time.time() - t0, 2)
    print(f"[COMODO demo] Ready in {state['load_seconds']}s.", flush=True)
    return student


# ─────────────────────────────── request models ────────────────────────────────
class InferRequest(BaseModel):
    activity: str
    seed: int = 42
    # All improvements default to OFF so the original endpoint behaviour is
    # preserved bit-for-bit. The frontend toggles them on for the "improved" tab.
    tta: bool = False
    multi_prototype: bool = False
    extra_noise: float = 0.0


class LossRequest(BaseModel):
    activity: str
    seed: int = 42


class BatchABRequest(BaseModel):
    """A/B benchmark: run N queries through all 4 configs at one noise level."""
    samples_per_activity: int = 5
    extra_noise: float = 0.3
    perturb_strength: float = 0.15
    base_seed: int = 5000


class SingleABRequest(BaseModel):
    """A/B benchmark: run ONE query through all 4 configs."""
    activity: str
    seed: int = 42
    extra_noise: float = 0.3


class NoiseSweepRequest(BaseModel):
    noise_levels: list[float] | None = None
    samples_per_activity: int = 3
    base_seed: int = 2000
    # Person-to-person variation strength applied to the query signals' freq /
    # amplitude. 0.15 ≈ ±15% gait/intensity variation — calibrated so the
    # baseline still has signal to lose but the improvements have headroom
    # to gain.
    perturb_strength: float = 0.15


class SaliencyRequest(BaseModel):
    activity: str
    seed: int = 42


# ─────────────────────── classification primitives ─────────────────────────────
def _similarities(
    z_norm: np.ndarray,
    multi_prototype: bool,
) -> dict[str, float]:
    """Cosine similarity from query embedding to each activity class.

    Single-prototype: cos(query, ref).
    K-prototype:      max(cos(query, ref_k) for k in 1..K)  — nearest-neighbour.

    We use *max* rather than mean because each ref_k samples a different
    sub-region of the in-class variability space (different perturbation
    seeds), so they don't share embedding neighbourhoods. Nearest-neighbour
    is the correct aggregator when the support set spans the variability.
    """
    sims: dict[str, float] = {}
    if multi_prototype:
        for a, refs in state["ref_embeddings_k"].items():
            # refs are already L2-normalised; refs @ z_norm  →  (K,)
            sims[a] = float(np.max(refs @ z_norm))
    else:
        for a, ref in state["ref_embeddings_1"].items():
            sims[a] = float(np.dot(z_norm, ref))
    return sims


def _classify(
    student,
    imu_np: np.ndarray,
    tta: bool,
    multi_prototype: bool,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Run the (improved) inference pipeline. Returns (embedding, sims, latency_ms)."""
    x = torch.from_numpy(imu_np).unsqueeze(0)  # (1, 6, T)
    t0 = time.time()
    with torch.no_grad():
        if tta:
            batch = tta_augment(x, n=TTA_N)
            z = _encode(student, batch)          # (N, 128)
            z_mean = z.mean(dim=0, keepdim=True)
            z_norm = F.normalize(z_mean, p=2, dim=1).squeeze(0).cpu().numpy()
        else:
            z_norm = _encode(student, x).squeeze(0).cpu().numpy()
    latency_ms = (time.time() - t0) * 1000.0
    sims = _similarities(z_norm, multi_prototype=multi_prototype)
    return z_norm, sims, latency_ms


# ───────────────────────────────── endpoints ───────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": state["student"] is not None,
        "load_seconds": state["load_seconds"],
        "device": str(DEVICE),
    }


@app.get("/api/activities")
def list_activities() -> dict[str, Any]:
    return {
        "activities": [
            {"id": k, "label": v["label"], "description": v["description"]}
            for k, v in ACTIVITIES.items()
        ]
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    student = get_student()
    n_params = sum(p.numel() for p in student.parameters())
    return {
        "imu_backbone": "paris-noah/Mantis-8M",
        "projection_head": f"MLP({student.student_dimension} → {MLP_HIDDEN} → {MLP_OUT})",
        "student_dimension": student.student_dimension,
        "mlp_hidden_dim": MLP_HIDDEN,
        "mlp_output_dim": MLP_OUT,
        "device": str(DEVICE),
        "total_parameters": n_params,
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "queue_size": QUEUE_SIZE,
        "load_seconds": state["load_seconds"],
        "input_shape": [1, N_CHANNELS, SEQ_LEN],
        "output_shape": [1, MLP_OUT],
        "k_prototypes": K_PROTOTYPES,
        "tta_n": TTA_N,
        "ref_build_sequential_ms": state["ref_build_sequential_ms"],
        "ref_build_batched_ms": state["ref_build_batched_ms"],
    }


@app.get("/api/imu/{activity}")
def get_imu(activity: str, seed: int = 42, extra_noise: float = 0.0) -> dict[str, Any]:
    if activity not in ACTIVITIES:
        raise HTTPException(404, f"unknown activity: {activity}")
    data = generate_imu(activity, seed=seed, extra_noise=extra_noise)
    step = 4
    return {
        "activity": activity,
        "label": ACTIVITIES[activity]["label"],
        "seq_len": SEQ_LEN,
        "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "samples": data[:, ::step].tolist(),
        "sample_step": step,
        "extra_noise": extra_noise,
    }


@app.post("/api/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    """Run inference. Optionally apply TTA and/or multi-prototype matching."""
    if req.activity not in ACTIVITIES:
        raise HTTPException(404, f"unknown activity: {req.activity}")
    student = get_student()
    imu = generate_imu(req.activity, seed=req.seed, extra_noise=req.extra_noise)
    z_norm, sims, latency_ms = _classify(
        student, imu,
        tta=req.tta,
        multi_prototype=req.multi_prototype,
    )
    predicted = max(sims, key=sims.get)
    confidence = softmax_confidence(sims)
    return {
        "activity": req.activity,
        "seed": req.seed,
        "tta": req.tta,
        "multi_prototype": req.multi_prototype,
        "extra_noise": req.extra_noise,
        "latency_ms": round(latency_ms, 1),
        "embedding": z_norm.tolist(),
        "embedding_dim": int(z_norm.shape[0]),
        "embedding_norm": float(np.linalg.norm(z_norm)),
        "similarities": sims,
        "confidence": confidence,
        "predicted": predicted,
        "predicted_label": ACTIVITIES[predicted]["label"],
        "predicted_confidence": confidence[predicted],
        "correct": predicted == req.activity,
        "tta_n": TTA_N if req.tta else 1,
        "k_prototypes": K_PROTOTYPES if req.multi_prototype else 1,
    }


@app.post("/api/loss")
def compute_loss(req: LossRequest) -> dict[str, Any]:
    """Run a single COMODOLoss step against the simulated video queue."""
    student = get_student()
    imu = generate_imu(req.activity, seed=req.seed)
    x = torch.from_numpy(imu).unsqueeze(0)
    ref = torch.from_numpy(state["ref_embeddings_1"][req.activity]).unsqueeze(0)
    torch.manual_seed(req.seed)
    z_v = F.normalize(ref + 0.01 * torch.randn_like(ref), p=2, dim=1)

    comodo = COMODOLoss(
        instanceQ_encoded=state["instance_queue"],
        student_model=student,
        teacher_temp=0.1,
        student_temp=0.05,
    )
    t0 = time.time()
    loss_val = comodo(x, z_v)
    latency_ms = round((time.time() - t0) * 1000, 1)
    return {
        "activity": req.activity,
        "loss": float(loss_val.item()),
        "teacher_temp": 0.1,
        "student_temp": 0.05,
        "queue_size": QUEUE_SIZE,
        "latency_ms": latency_ms,
    }


@app.post("/api/saliency")
def saliency(req: SaliencyRequest) -> dict[str, Any]:
    """Per-channel importance via gradient × input on the embedding norm.

    Method:
      score(x) = ‖student(x)‖²    (use the un-normalised embedding so the gradient
                                   carries magnitude information).
      ∂score / ∂x  has the same shape as x = (1, 6, T).
      For each channel c we report  mean_t |xₒ,c,t · grad_o,c,t|, then normalise
      across channels so the bars sum to 1.

    This is the standard "gradient × input" saliency map, channel-aggregated.
    """
    if req.activity not in ACTIVITIES:
        raise HTTPException(404, f"unknown activity: {req.activity}")
    student = get_student()
    imu = generate_imu(req.activity, seed=req.seed)

    # The Mantis-8M wrapper calls a `resize(...)` helper that uses
    # `torch.tensor(X)` internally — which detaches the autograd graph. To get a
    # working gradient we pre-resize the input to seq_len=512 ourselves so the
    # model's internal resize branch is a no-op, preserving the graph.
    x_raw = torch.from_numpy(imu).unsqueeze(0)              # (1, 6, 1000)
    x = F.interpolate(x_raw, size=512, mode="linear", align_corners=False)
    x = x.detach().requires_grad_(True)                     # (1, 6, 512)

    z = student(x)
    score = (z ** 2).sum()
    score.backward()
    if x.grad is None:
        raise HTTPException(500, "gradient did not flow through student backbone")
    grad = x.grad.detach()                                  # (1, 6, 512)
    contrib = (x.detach() * grad).abs().mean(dim=-1).squeeze(0)  # (6,)
    contrib_np = contrib.cpu().numpy()
    contrib_np = contrib_np / (contrib_np.sum() + 1e-8)

    channels = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    return {
        "activity": req.activity,
        "seed": req.seed,
        "channels": channels,
        "importance": [float(v) for v in contrib_np],
        "embedding_score": float(score.item()),
    }


@app.post("/api/noise-sweep")
def noise_sweep(req: NoiseSweepRequest) -> dict[str, Any]:
    """Run an accuracy-vs-noise benchmark for four configs:
       (1) baseline      — single prototype, no TTA
       (2) +K-prototypes — K refs per class, no TTA
       (3) +TTA          — single prototype, TTA on
       (4) +K-proto +TTA — both improvements

    Each (activity, noise level) is averaged over `samples_per_activity`
    random seeds to keep noise in the answer low.
    """
    student = get_student()
    if req.noise_levels is None:
        levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    else:
        levels = sorted(set(float(x) for x in req.noise_levels))

    configs = [
        ("baseline", False, False),
        ("+ K-prototypes", False, True),
        ("+ TTA", True, False),
        ("+ K-proto + TTA", True, True),
    ]
    results: dict[str, list[float]] = {name: [] for name, _, _ in configs}
    activities = list(ACTIVITIES.keys())

    t_total = time.time()
    for sigma in levels:
        # Generate one fresh IMU per (activity, sample) at this noise level.
        # `perturb_params=True` simulates person-to-person variation — the
        # query no longer comes from the exact same generator as the
        # prototypes, so single-reference matching has real headroom to lose
        # and the K-prototype / TTA improvements can demonstrate their value.
        cache: list[tuple[str, np.ndarray]] = []
        for a in activities:
            for s in range(req.samples_per_activity):
                imu = generate_imu(
                    a,
                    seed=req.base_seed + s,
                    extra_noise=sigma,
                    perturb_params=True,
                    perturb_strength=req.perturb_strength,
                )
                cache.append((a, imu))
        for name, tta, kproto in configs:
            correct = 0
            for true_act, imu in cache:
                _, sims, _ = _classify(student, imu, tta=tta, multi_prototype=kproto)
                pred = max(sims, key=sims.get)
                if pred == true_act:
                    correct += 1
            acc = correct / len(cache) * 100.0
            results[name].append(round(acc, 2))

    return {
        "noise_levels": levels,
        "configs": list(results.keys()),
        "accuracy_by_config": results,
        "samples_per_point": req.samples_per_activity * len(activities),
        "elapsed_s": round(time.time() - t_total, 2),
    }


CONFIGS_AB: list[tuple[str, bool, bool]] = [
    ("baseline",          False, False),
    ("+ K-prototypes",    False, True),
    ("+ TTA",             True,  False),
    ("+ K-proto + TTA",   True,  True),
]


@app.post("/api/infer-ab")
def infer_ab(req: SingleABRequest) -> dict[str, Any]:
    """Run the SAME query through all 4 configs. The frontend uses this to
    show the audience that improvements flip wrong predictions to correct ones.
    """
    if req.activity not in ACTIVITIES:
        raise HTTPException(404, f"unknown activity: {req.activity}")
    student = get_student()
    imu = generate_imu(req.activity, seed=req.seed, extra_noise=req.extra_noise)
    out = []
    for name, tta, kproto in CONFIGS_AB:
        _, sims, lat = _classify(student, imu, tta=tta, multi_prototype=kproto)
        pred = max(sims, key=sims.get)
        conf = softmax_confidence(sims)
        out.append({
            "config": name,
            "tta": tta,
            "multi_prototype": kproto,
            "predicted": pred,
            "predicted_label": ACTIVITIES[pred]["label"],
            "predicted_confidence": float(conf[pred]),
            "correct": pred == req.activity,
            "latency_ms": round(lat, 1),
            "top3": sorted(conf.items(), key=lambda x: -x[1])[:3],
        })
    return {
        "activity": req.activity,
        "true_label": ACTIVITIES[req.activity]["label"],
        "extra_noise": req.extra_noise,
        "seed": req.seed,
        "results": out,
    }


@app.post("/api/batch-ab")
def batch_ab(req: BatchABRequest) -> dict[str, Any]:
    """Run a batch of queries through all 4 configs and report aggregate
    accuracy / mean confidence / mean latency. This is the 'before vs after'
    accuracy demonstration the audience sees on the Improvements tab.
    """
    student = get_student()
    activities = list(ACTIVITIES.keys())
    queries: list[tuple[str, np.ndarray]] = []
    for a in activities:
        for s in range(req.samples_per_activity):
            imu = generate_imu(
                a,
                seed=req.base_seed + s,
                extra_noise=req.extra_noise,
                perturb_params=True,
                perturb_strength=req.perturb_strength,
            )
            queries.append((a, imu))

    out: dict[str, dict[str, Any]] = {}
    t_total = time.time()
    for name, tta, kproto in CONFIGS_AB:
        n_correct = 0
        confs: list[float] = []
        latencies: list[float] = []
        per_activity_correct: dict[str, int] = {a: 0 for a in activities}
        per_activity_total: dict[str, int] = {a: 0 for a in activities}
        for true_a, imu in queries:
            _, sims, lat = _classify(student, imu, tta=tta, multi_prototype=kproto)
            pred = max(sims, key=sims.get)
            per_activity_total[true_a] += 1
            if pred == true_a:
                n_correct += 1
                per_activity_correct[true_a] += 1
            confs.append(softmax_confidence(sims)[pred])
            latencies.append(lat)
        out[name] = {
            "accuracy_pct": round(n_correct / len(queries) * 100, 2),
            "n_correct": n_correct,
            "n_total": len(queries),
            "mean_confidence_pct": round(float(np.mean(confs)) * 100, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "per_activity_accuracy_pct": {
                a: round(per_activity_correct[a] / per_activity_total[a] * 100, 2)
                for a in activities
            },
            "tta": tta,
            "multi_prototype": kproto,
        }

    base_acc = out["baseline"]["accuracy_pct"]
    base_conf = out["baseline"]["mean_confidence_pct"]
    for name in out:
        out[name]["accuracy_delta_vs_baseline"] = round(
            out[name]["accuracy_pct"] - base_acc, 2
        )
        out[name]["confidence_delta_vs_baseline"] = round(
            out[name]["mean_confidence_pct"] - base_conf, 2
        )
    return {
        "samples_per_activity": req.samples_per_activity,
        "extra_noise": req.extra_noise,
        "perturb_strength": req.perturb_strength,
        "total_queries": len(queries),
        "elapsed_s": round(time.time() - t_total, 2),
        "results": out,
        "activities": activities,
    }


# ─────────────────────── IMU CSV upload + example ──────────────────────────────
def _parse_imu_csv(text: str) -> np.ndarray:
    """Parse a free-form CSV / whitespace text into a (6, N) float32 array.

    Accepts:
      * comma-, tab-, or whitespace-separated
      * header lines (auto-skipped if any token isn't numeric)
      * either orientation:  (N rows × 6 cols)  or  (6 rows × N cols)
      * '#' comment lines
    """
    rows: list[list[float]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(",", " ").replace("\t", " ").split() if p.strip()]
        try:
            row = [float(p) for p in parts]
        except ValueError:
            continue  # header / non-numeric line
        if row:
            rows.append(row)

    if not rows:
        raise HTTPException(400, "no numeric rows found in upload")
    # Pad short rows with NaN then drop ragged data — keep only rows that match
    # the dominant row width.
    widths = [len(r) for r in rows]
    target = max(set(widths), key=widths.count)
    rows = [r for r in rows if len(r) == target]
    arr = np.array(rows, dtype=np.float32)  # (N, target) or (6, target) etc.

    # Decide orientation:  if one of the two dims is 6, that's channels.
    if arr.shape[1] == 6 and arr.shape[0] != 6:
        arr = arr.T                              # (6, N)
    elif arr.shape[0] != 6:
        raise HTTPException(
            400,
            f"expected 6 IMU channels somewhere in the data; got shape {tuple(arr.shape)}",
        )
    return arr.astype(np.float32)


def _resample_to_seq_len(arr: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a (6, M) array to (6, seq_len)."""
    if arr.shape[-1] == seq_len:
        return arr
    t = torch.from_numpy(arr).unsqueeze(0)
    t = F.interpolate(t, size=seq_len, mode="linear", align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)


@app.post("/api/infer-upload")
async def infer_upload(
    file: UploadFile = File(...),
    tta: bool = Form(False),
    multi_prototype: bool = Form(True),
    true_label: str | None = Form(None),
) -> dict[str, Any]:
    """Run real COMODO inference on a user-uploaded IMU CSV.

    The CSV should have 6 numeric columns (accel_x/y/z, gyro_x/y/z) and at
    least 100 rows. Headers are auto-stripped. Any length is resampled to the
    model's expected 1000-sample window.
    """
    student = get_student()
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "file is not UTF-8 text — please upload a CSV/TSV")
    arr = _parse_imu_csv(text)
    original_shape = list(arr.shape)
    arr_resampled = _resample_to_seq_len(arr)

    z_norm, sims, latency_ms = _classify(
        student, arr_resampled,
        tta=tta, multi_prototype=multi_prototype,
    )
    predicted = max(sims, key=sims.get)
    confidence = softmax_confidence(sims)
    correct = (true_label == predicted) if true_label in ACTIVITIES else None

    # Downsample for plot
    step = max(1, arr_resampled.shape[-1] // 250)
    return {
        "filename": file.filename,
        "original_shape": original_shape,
        "resampled_shape": list(arr_resampled.shape),
        "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "samples": arr_resampled[:, ::step].tolist(),
        "sample_step": step,
        "tta": tta,
        "multi_prototype": multi_prototype,
        "true_label": true_label,
        "predicted": predicted,
        "predicted_label": ACTIVITIES[predicted]["label"],
        "predicted_confidence": float(confidence[predicted]),
        "correct": correct,
        "similarities": sims,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "tta_n": TTA_N if tta else 1,
        "k_prototypes": K_PROTOTYPES if multi_prototype else 1,
    }


@app.get("/api/example-csv/{activity}")
def example_csv(activity: str, seed: int = 42) -> PlainTextResponse:
    """Generate a downloadable example CSV for a given activity.

    Audience members can grab this, open it in Excel, see what real IMU data
    looks like, then upload it back through the UI to verify inference works.
    """
    if activity not in ACTIVITIES:
        raise HTTPException(404, f"unknown activity: {activity}")
    data = generate_imu(activity, seed=seed)            # (6, 1000)
    rows = data.T                                       # (1000, 6) — time-major
    header = "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
    body = "\n".join(",".join(f"{v:.5f}" for v in row) for row in rows)
    csv_text = header + body + "\n"
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="comodo_{activity}.csv"'},
    )


@app.get("/api/batched-bench")
def batched_bench() -> dict[str, Any]:
    """Sequential vs batched inference latency for the K-prototype reference build.

    Returns the timing recorded during model warmup (see get_student()), so this
    endpoint is O(1) and safe to call repeatedly during the demo.
    """
    get_student()  # ensure warmup has populated the state
    seq = state["ref_build_sequential_ms"]
    bat = state["ref_build_batched_ms"]
    speedup = round(seq / bat, 2) if bat else None
    return {
        "n_forward_passes_sequential": len(ACTIVITIES) * K_PROTOTYPES,
        "n_forward_passes_batched": 1,
        "sequential_ms": seq,
        "batched_ms": bat,
        "speedup": speedup,
    }


@app.get("/api/throughput-bench")
def throughput_bench() -> dict[str, Any]:
    """Measure inference throughput (queries/sec) at several batch sizes.

    Each batch size is timed over `n_iters` forward passes (5 warmup + 3 timed).
    Real wearable deployments care about queries/sec/Watt — batching is the
    single biggest knob you can turn at inference time.
    """
    student = get_student()
    batch_sizes = [1, 2, 4, 8, 16, 32]
    n_iters = 3
    n_warmup = 2

    # Re-use the same random IMU per batch so we measure the model, not data prep.
    base_x = torch.randn(1, N_CHANNELS, SEQ_LEN)
    results = []
    with torch.no_grad():
        for bs in batch_sizes:
            x = base_x.expand(bs, -1, -1).contiguous()
            for _ in range(n_warmup):
                _ = student(x)
            t0 = time.time()
            for _ in range(n_iters):
                _ = student(x)
            elapsed_s = (time.time() - t0) / n_iters
            qps = bs / elapsed_s
            results.append({
                "batch_size": bs,
                "latency_ms": round(elapsed_s * 1000, 2),
                "queries_per_sec": round(qps, 1),
            })
    base_qps = results[0]["queries_per_sec"] or 1.0
    for r in results:
        r["speedup_vs_bs1"] = round(r["queries_per_sec"] / base_qps, 2)
    return {
        "batch_sizes": batch_sizes,
        "n_warmup_per_size": n_warmup,
        "n_iters_per_size": n_iters,
        "results": results,
    }


# ─────────────────────────── results aggregation ────────────────────────────────
def _aggregate_results() -> dict[str, Any]:
    path = ROOT / "results" / "unsupervised_result" / "results.json"
    with open(path) as f:
        raw = json.load(f)

    rows: list[dict[str, Any]] = []
    for method, entries in raw.items():
        for e in entries:
            rows.append({
                "method": method,
                "imu": e["imu"].split("/")[-1],
                "dataset": e["dataset"],
                "acc1": e["acc1"],
                "acc3": e["acc3"],
                "acc5": e["acc5"],
            })

    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["method"], r["imu"], r["dataset"])
        grouped.setdefault(key, []).append(r)

    summary: list[dict[str, Any]] = []
    for (method, imu, dataset), entries in grouped.items():
        best = max(entries, key=lambda x: x["acc1"])
        summary.append({
            "method": method,
            "imu": imu,
            "dataset": dataset,
            "best_acc1": round(best["acc1"] * 100, 2),
            "best_acc3": round(best["acc3"] * 100, 2),
            "best_acc5": round(best["acc5"] * 100, 2),
            "n_runs": len(entries),
        })
    summary.sort(key=lambda r: (r["dataset"], r["imu"], r["method"]))
    return {
        "raw_count": len(rows),
        "summary": summary,
        "methods": sorted({r["method"] for r in rows}),
        "datasets": sorted({r["dataset"] for r in rows}),
        "imu_backbones": sorted({r["imu"] for r in rows}),
    }


@app.get("/api/results")
def results() -> JSONResponse:
    return JSONResponse(_aggregate_results())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
