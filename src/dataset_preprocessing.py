import os
import glob
import numpy as np

HAND_IDX  = slice(3, 20)  
CHEST_IDX = slice(20, 37)  
ANKLE_IDX = slice(37, 54)  
HR_IDX    = 2              
LABEL_IDX = 1              
TIME_IDX  = 0              

DESIRED_IMU_DIM = 64       
TARGET_HZ = 25             
WIN_SEC = 5.0            
HOP_SEC = 2.5            
RNG_SEED = 42             
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15 

RAW_DIR = "./data/pamap2"  
OUT_DIR = "./data"         

def load_subject(path):
    """
    Load a PAMAP2 subject .dat file (whitespace-separated).
    Returns (t, label, hr, hand, chest, ankle) as float arrays where possible.
    """
    arr = np.genfromtxt(path)
    if arr.ndim == 1: 
        arr = arr[None, :]
    t = arr[:, TIME_IDX].astype(np.float64)
    y = arr[:, LABEL_IDX].astype(np.int64)
    hr = arr[:, HR_IDX].astype(np.float32)

    hand  = arr[:, HAND_IDX].astype(np.float32)
    chest = arr[:, CHEST_IDX].astype(np.float32)
    ankle = arr[:, ANKLE_IDX].astype(np.float32)


    keep = np.ones_like(t, dtype=bool)
    keep[1:] = t[1:] > t[:-1]
    if not keep.all():
        t, y, hr, hand, chest, ankle = t[keep], y[keep], hr[keep], hand[keep], chest[keep], ankle[keep]
    return t, y, hr, hand, chest, ankle

def resample_to_fixed_rate(t, x, target_hz):
    """
    Resample per-channel signal x(t) to uniform grid at target_hz using linear interpolation.
    - t: (T,) seconds, strictly increasing
    - x: (T, C)
    Returns:
      t_u: (Tu,), x_u: (Tu, C)
    """
    if len(t) == 0:
        return np.array([]), np.zeros((0, x.shape[1]), dtype=x.dtype)
    t0, t1 = t[0], t[-1]
    if t1 <= t0:
        return np.array([]), np.zeros((0, x.shape[1]), dtype=x.dtype)
    step = 1.0 / float(target_hz)
    t_u = np.arange(t0, t1 + 1e-6, step, dtype=np.float64)

    # Handle NaNs by linear interpolation per channel; edge NaNs are forward/back filled
    x_u = np.empty((t_u.shape[0], x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        xc = x[:, c]
        valid = ~np.isnan(xc)
        if valid.sum() < 2:
            x_u[:, c] = 0.0
            continue
        x_u[:, c] = np.interp(t_u, t[valid], xc[valid], left=xc[valid][0], right=xc[valid][-1]).astype(np.float32)
    return t_u, x_u

def pad_or_truncate(x, out_dim):
    """
    x: (T, C_in) -> (T, out_dim) by zero-padding or truncating channels.
    """
    T, C = x.shape
    if C == out_dim:
        return x
    if C > out_dim:
        return x[:, :out_dim]
    out = np.zeros((T, out_dim), dtype=x.dtype)
    out[:, :C] = x
    return out

def window_indices(T, win, hop):
    idxs = []
    s = 0
    while s + win <= T:
        idxs.append((s, s + win))
        s += hop
    return idxs

def majority_label(y_slice):
    vals, counts = np.unique(y_slice, return_counts=True)
    return vals[np.argmax(counts)]

def build_windows(hr, hand, chest, ankle, y, win, hop):
    """
    Inputs all sampled at same 25 Hz timeline.
    Shapes:
      hr:    (Tu, 1)
      hand:  (Tu, C1)
      chest: (Tu, C2)
      ankle: (Tu, C3)
      y:     (Tu,)
    Returns stacked window datasets for each modality and labels.
    """
    T = y.shape[0]
    idxs = window_indices(T, win, hop)
    Xh, Xc, Xa, Xhr, Y = [], [], [], [], []
    for s, e in idxs:
        Xh.append(hand[s:e])
        Xc.append(chest[s:e])
        Xa.append(ankle[s:e])
        Xhr.append(hr[s:e])
        Y.append(majority_label(y[s:e]))
    if len(Y) == 0:
        return (np.zeros((0, win, hand.shape[1]), dtype=np.float32),
                np.zeros((0, win, chest.shape[1]), dtype=np.float32),
                np.zeros((0, win, ankle.shape[1]), dtype=np.float32),
                np.zeros((0, win, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.stack(Xh), np.stack(Xc), np.stack(Xa), np.stack(Xhr), np.asarray(Y, dtype=np.int64)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subject_paths = sorted(glob.glob(os.path.join(RAW_DIR, "subject*.dat")))
    if not subject_paths:
        raise FileNotFoundError(f"No subject*.dat files found in {RAW_DIR}")

    all_hand, all_chest, all_ankle, all_hr, all_y = [], [], [], [], []

    for p in subject_paths:
        print(f"Loading {os.path.basename(p)}")
        t, y, hr, hand, chest, ankle = load_subject(p)

        tu, hr_u    = resample_to_fixed_rate(t, hr[:, None], TARGET_HZ)
        _,  hand_u  = resample_to_fixed_rate(t, hand, TARGET_HZ)
        _,  chest_u = resample_to_fixed_rate(t, chest, TARGET_HZ)
        _,  ankle_u = resample_to_fixed_rate(t, ankle, TARGET_HZ)
        _,  y_u     = resample_to_fixed_rate(t, y[:, None].astype(np.float32), TARGET_HZ)
        y_u = y_u[:, 0].round().astype(np.int64)

        hr_u = np.nan_to_num(hr_u, nan=0.0)

        hand_u  = pad_or_truncate(hand_u,  DESIRED_IMU_DIM)
        chest_u = pad_or_truncate(chest_u, DESIRED_IMU_DIM)
        ankle_u = pad_or_truncate(ankle_u, DESIRED_IMU_DIM)

        WIN = int(WIN_SEC * TARGET_HZ)
        HOP = int(HOP_SEC * TARGET_HZ)
        Xh, Xc, Xa, Xhr, Y = build_windows(hr_u, hand_u, chest_u, ankle_u, y_u, WIN, HOP)

        if len(Y) > 0:
            all_hand.append(Xh)
            all_chest.append(Xc)
            all_ankle.append(Xa)
            all_hr.append(Xhr)
            all_y.append(Y)

    if not all_y:
        raise RuntimeError("No windows produced. Check column indices and sampling params.")

    imu_hand   = np.concatenate(all_hand, axis=0).astype(np.float32)
    imu_chest  = np.concatenate(all_chest, axis=0).astype(np.float32)
    imu_ankle  = np.concatenate(all_ankle, axis=0).astype(np.float32)
    heart_rate = np.concatenate(all_hr, axis=0).astype(np.float32)
    labels     = np.concatenate(all_y, axis=0).astype(np.int64)

    N = len(labels)
    assert imu_hand.shape[0] == imu_chest.shape[0] == imu_ankle.shape[0] == heart_rate.shape[0] == N

    
    rng = np.random.default_rng(RNG_SEED)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * N)
    n_val   = int(VAL_FRAC * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    def write_split(name, sel):
        d = os.path.join(OUT_DIR, name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "imu_hand.npy"),   imu_hand[sel])
        np.save(os.path.join(d, "imu_chest.npy"),  imu_chest[sel])
        np.save(os.path.join(d, "imu_ankle.npy"),  imu_ankle[sel])
        np.save(os.path.join(d, "heart_rate.npy"), heart_rate[sel])
        np.save(os.path.join(d, "labels.npy"),     labels[sel])

    write_split("train", train_idx)
    write_split("val",   val_idx)
    write_split("test",  test_idx)

    print("Done. Wrote:")
    for split in ["train", "val", "test"]:
        base = os.path.join(OUT_DIR, split)
        print(f"  {split}:",
              os.path.join(base, "imu_hand.npy"),
              os.path.join(base, "imu_chest.npy"),
              os.path.join(base, "imu_ankle.npy"),
              os.path.join(base, "heart_rate.npy"),
              os.path.join(base, "labels.npy"))

if __name__ == "__main__":
    main()
