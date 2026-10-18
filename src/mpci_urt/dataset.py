import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _trajectory_columns(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(
            f"No trajectory-score columns found with prefix '{prefix}'."
        )

    def key(col):
        suffix = col[len(prefix):]
        return int(suffix) if suffix.isdigit() else suffix

    return sorted(cols, key=key)


def load_arrays(config):
    data_cfg = config["data"]
    pre_cfg = config["preprocessing"]

    trips = pd.read_csv(data_cfg["trips"])
    topology = pd.read_csv(data_cfg["topology"])
    trajectory_df = pd.read_csv(data_cfg["trajectory_scores"])

    required = data_cfg["feature_columns"] + [data_cfg["label_column"]]
    missing = [c for c in required if c not in trips.columns]
    if missing:
        raise ValueError(f"Missing trip columns: {missing}")

    if not {"origin", "destination"}.issubset(topology.columns):
        raise ValueError(
            "Topology file must contain origin and destination columns."
        )

    traj_cols = _trajectory_columns(
        trajectory_df,
        data_cfg.get("trajectory_prefix", "dest_"),
    )

    if len(trips) != len(trajectory_df):
        raise ValueError(
            "trips and trajectory_scores must have identical row counts "
            "and row order."
        )

    x_raw = trips[data_cfg["feature_columns"]].to_numpy(
        dtype=np.float32
    )
    y = trips[data_cfg["label_column"]].to_numpy(dtype=np.int64)
    trajectory_scores = trajectory_df[traj_cols].to_numpy(
        dtype=np.float32
    )

    edge_index = torch.tensor(
        topology[["origin", "destination"]]
        .to_numpy(dtype=np.int64)
        .T,
        dtype=torch.long,
    )

    split_idx = int(
        len(x_raw) * float(pre_cfg.get("train_ratio", 0.8))
    )
    if split_idx <= 0 or split_idx >= len(x_raw):
        raise ValueError(
            "train_ratio produces an empty train or test split."
        )

    scaler = MinMaxScaler()
    fit_scope = pre_cfg.get(
        "scaler_fit_scope", "train"
    ).lower()

    if fit_scope == "all":
        x = scaler.fit_transform(x_raw).astype(np.float32)
    elif fit_scope == "train":
        scaler.fit(x_raw[:split_idx])
        x = scaler.transform(x_raw).astype(np.float32)
    else:
        raise ValueError(
            "scaler_fit_scope must be 'all' or 'train'."
        )

    num_classes = trajectory_scores.shape[1]
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError(
            f"Destination labels range from {y.min()} to {y.max()}, "
            f"but trajectory scores contain {num_classes} classes."
        )

    return {
        "x_train": x[:split_idx],
        "y_train": y[:split_idx],
        "traj_train": trajectory_scores[:split_idx],
        "x_test": x[split_idx:],
        "y_test": y[split_idx:],
        "traj_test": trajectory_scores[split_idx:],
        "edge_index": edge_index,
        "num_features": x.shape[1],
        "num_classes": num_classes,
    }


def build_loader(
    x,
    y,
    trajectory_scores,
    edge_index,
    seq_len,
    batch_size,
    shuffle,
):
    graphs = []

    for i in range(0, len(x) - seq_len + 1, seq_len):
        graphs.append(
            Data(
                x=torch.tensor(
                    x[i:i + seq_len],
                    dtype=torch.float32,
                ),
                edge_index=edge_index.clone(),
                y=torch.tensor(
                    y[i:i + seq_len],
                    dtype=torch.long,
                ),
                trajectory_scores=torch.tensor(
                    trajectory_scores[i:i + seq_len],
                    dtype=torch.float32,
                ),
            )
        )

    if not graphs:
        raise ValueError(
            f"Not enough records for seq_len={seq_len}."
        )

    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
