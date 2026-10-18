import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpci_urt.config import load_config
from mpci_urt.dataset import load_arrays, build_loader
from mpci_urt.model import MPCIURT
from mpci_urt.trainer import resolve_device, set_seed, evaluate_model

def main(config_path):
    config = load_config(config_path)
    set_seed(int(config["experiment"]["seed"]))
    device = resolve_device(config["experiment"].get("device", "auto"))
    print(f"Device: {device}")

    arrays = load_arrays(config)
    seq_len = int(config["preprocessing"]["seq_len"])
    test_batch_size = max(1, int(len(arrays["x_test"]) / seq_len))

    test_loader = build_loader(
        arrays["x_test"],
        arrays["y_test"],
        arrays["traj_test"],
        arrays["edge_index"],
        seq_len=seq_len,
        batch_size=test_batch_size,
        shuffle=False,
    )

    m = config["model"]
    model = MPCIURT(
        input_dim=arrays["num_features"],
        output_dim=arrays["num_classes"],
        gnn_hidden_1=int(m["gnn_hidden_1"]),
        gnn_hidden_2=int(m["gnn_hidden_2"]),
        lstm_hidden=int(m["lstm_hidden"]),
        lstm_layers=int(m["lstm_layers"]),
        fusion_hidden=int(m["fusion_hidden"]),
    )

    checkpoint = torch.load(
        config["output"]["checkpoint"], map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(
        model=model,
        test_loader=test_loader,
        edge_index=arrays["edge_index"],
        device=device,
        predictions_path=config["output"]["predictions"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
