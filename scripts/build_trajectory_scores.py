import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mpci_urt.config import load_config
from mpci_urt.trajectory import (
    build_subway_network,
    combine_station_times,
    estimate_transfer_times,
    estimate_waiting_times,
    estimate_walking_times,
    generate_trajectory_scores_mnlogit,
    graph_edges_dataframe,
    prepare_candidate_paths,
)


def main(config_path):
    config = load_config(config_path)
    tcfg = config["trajectory_generation"]

    trips = pd.read_csv(tcfg["trips"])
    stations = pd.read_csv(tcfg["stations"])
    timetable = pd.read_csv(tcfg["timetable"])
    base_stations = pd.read_csv(tcfg["base_stations"])
    transfer_links = pd.read_csv(tcfg["transfer_links"])

    timetable["arrival_time"] = pd.to_datetime(timetable["arrival_time"])
    timetable["departure_time"] = pd.to_datetime(timetable["departure_time"])

    circular_line_id = tcfg.get("circular_line_id")
    circular_period = tcfg.get("circular_line_period_seconds")

    walking = estimate_walking_times(
        trips,
        stations,
        timetable,
        circular_line_id=circular_line_id,
        circular_line_period_seconds=circular_period,
    )
    waiting = estimate_waiting_times(trips, stations)
    transfer = estimate_transfer_times(
        trips,
        stations,
        timetable,
        base_stations,
        walking,
        transfer_station_value=tcfg.get(
            "transfer_station_value", "transfer"
        ),
    )
    station_times = combine_station_times(
        stations, walking, waiting, transfer
    )

    graph = build_subway_network(
        stations,
        timetable,
        transfer_links,
        base_stations,
        circular_line_id=circular_line_id,
        circular_line_period_seconds=circular_period,
    )
    topology = graph_edges_dataframe(graph)

    candidate_paths = prepare_candidate_paths(
        graph=graph,
        trips=trips,
        station_times=station_times,
        candidate_destinations=int(tcfg.get("candidate_destinations", 3)),
        paths_per_destination=int(tcfg.get("paths_per_destination", 3)),
        best_candidate_mode=tcfg.get(
            "best_candidate_mode", "legacy_max_destination_id"
        ),
    )

    # Use every public station ID as a destination class so that
    # trajectory-score columns align exactly with model labels.
    destination_classes = sorted(
        stations["station_id"].astype(int).unique().tolist()
    )
    scores, score_meta, fitted = generate_trajectory_scores_mnlogit(
        candidate_paths, destination_classes
    )

    for path_key in [
        "station_times_output",
        "topology_output",
        "candidate_paths_output",
        "trajectory_scores_output",
        "trajectory_metadata_output",
    ]:
        Path(tcfg[path_key]).parent.mkdir(parents=True, exist_ok=True)

    station_times.to_csv(tcfg["station_times_output"], index=False)
    topology.to_csv(tcfg["topology_output"], index=False)
    candidate_paths.to_csv(tcfg["candidate_paths_output"], index=False)
    scores.to_csv(tcfg["trajectory_scores_output"], index=False)
    score_meta.to_csv(tcfg["trajectory_metadata_output"], index=False)

    print(fitted.summary())
    print("Trajectory-score generation complete.")
    print(f"Scores: {tcfg['trajectory_scores_output']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
