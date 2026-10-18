import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpci_urt.config import load_config
from mpci_urt.dataset import load_arrays, build_loader
from mpci_urt.model import MPCIURT
from mpci_urt.trainer import resolve_device, set_seed, train_model

def main(config_path):
    config = load_config(config_path)
    set_seed(int(config["experiment"]["seed"]))
    device = resolve_device(config["experiment"].get("device", "auto"))
    print(f"Device: {device}")

    arrays = load_arrays(config)
    seq_len = int(config["preprocessing"]["seq_len"])
    batch_size = int(config["training"]["batch_size"])

    train_loader = build_loader(
        arrays["x_train"],
        arrays["y_train"],
        arrays["traj_train"],
        arrays["edge_index"],
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
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

    t = config["training"]
    train_model(
        model=model,
        train_loader=train_loader,
        edge_index=arrays["edge_index"],
        device=device,
        epochs=int(t["epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        alpha=float(t["alpha"]),
        checkpoint_path=config["output"]["checkpoint"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
