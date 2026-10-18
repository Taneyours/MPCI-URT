from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _timetable_seconds(temp_timetable, origin_name, destination_name):
    """Return timetable travel time in seconds using the ordering logic of the source script."""
    origin_idx = temp_timetable[temp_timetable["station_name"] == origin_name].index[0]
    destination_idx = temp_timetable[
        temp_timetable["station_name"] == destination_name
    ].index[0]

    if origin_idx < destination_idx:
        seconds = (
            temp_timetable.at[destination_idx, "arrival_time"]
            - temp_timetable.at[origin_idx, "departure_time"]
        ).total_seconds()
    else:
        seconds = (
            temp_timetable.at[origin_idx, "arrival_time"]
            - temp_timetable.at[destination_idx, "departure_time"]
        ).total_seconds()
    return float(seconds)


def estimate_walking_times(
    trips,
    stations,
    timetable,
    circular_line_id=None,
    circular_line_period_seconds=None,
):
    """Estimate station entry/exit walking times following the supplied source algorithm."""
    result = stations.copy()
    result["shortest_walking_time"] = 0.0

    for station_id in result["station_id"]:
        line = result.loc[result["station_id"] == station_id, "line_id"].iloc[0]
        line_stations = result[result["line_id"] == line].reset_index(drop=True)
        origin_trips = trips[trips["origin_station"] == station_id]

        walking_sum = 0.0
        counter = 0

        for destination_id in line_stations["station_id"]:
            if destination_id == station_id:
                continue

            choice = origin_trips[
                origin_trips["destination_station"] == destination_id
            ].sort_values("duration")

            if choice.empty:
                continue

            counter += 1
            shortest_travel_time = float(choice.iloc[0]["duration"])

            temp_timetable = timetable[timetable["line_id"] == line]
            if circular_line_id is not None and line == circular_line_id:
                temp_timetable = temp_timetable[temp_timetable["direction"] == 0]

            origin_name = result.loc[
                result["station_id"] == station_id, "station_name"
            ].iloc[0]
            destination_name = result.loc[
                result["station_id"] == destination_id, "station_name"
            ].iloc[0]

            subway_time = _timetable_seconds(
                temp_timetable, origin_name, destination_name
            )

            if (
                circular_line_id is not None
                and line == circular_line_id
                and circular_line_period_seconds is not None
                and subway_time >= circular_line_period_seconds / 2
            ):
                subway_time = circular_line_period_seconds - subway_time

            walking_sum += (shortest_travel_time - subway_time) / 2.0

        if counter > 0:
            if walking_sum < 0:
                walking_sum = -walking_sum
            result.loc[
                result["station_id"] == station_id, "shortest_walking_time"
            ] = walking_sum / counter

    return result


def estimate_waiting_times(trips, stations):
    """Estimate average station waiting time following the supplied source algorithm."""
    result = stations.copy()
    result["waiting_time"] = 0.0

    for origin_station in result["station_id"]:
        line = result.loc[
            result["station_id"] == origin_station, "line_id"
        ].iloc[0]
        line_stations = result[result["line_id"] == line]
        origin_trips = trips[trips["origin_station"] == origin_station]

        wait_sum = 0.0
        counter = 0

        for destination in line_stations["station_id"]:
            if destination == origin_station:
                continue

            choice = origin_trips[
                origin_trips["destination_station"] == destination
            ].sort_values("duration")

            if choice.empty:
                continue

            fastest = float(choice.iloc[0]["duration"])
            wait_sum += float((choice["duration"] - fastest).mean())
            counter += 1

        if counter > 0:
            result.loc[
                result["station_id"] == origin_station, "waiting_time"
            ] = wait_sum / counter

    return result


def estimate_transfer_times(
    trips,
    stations,
    timetable,
    base_stations,
    walking_times,
    transfer_station_value="transfer",
):
    """Estimate transfer time using the same decomposition as the supplied source script."""
    result = stations.copy()
    result["transfer_time"] = 0.0

    transfer_rows = result[result["station_type"] == transfer_station_value]

    for row_idx, row in transfer_rows.iterrows():
        transfer_name = row["station_name"]
        origin_reference = row["transfer_origin_name"]
        destination_reference = row["transfer_destination_name"]

        if pd.isna(origin_reference) or pd.isna(destination_reference):
            continue

        origin_lines = base_stations[
            base_stations["station_name"] == origin_reference
        ]["line_id"].tolist()
        transfer_lines = base_stations[
            base_stations["station_name"] == transfer_name
        ]["line_id"].tolist()
        destination_lines = base_stations[
            base_stations["station_name"] == destination_reference
        ]["line_id"].tolist()

        common_origin = [x for x in origin_lines if x in transfer_lines]
        common_destination = [x for x in destination_lines if x in transfer_lines]
        if not common_origin or not common_destination:
            continue

        line_o = common_origin[0]
        line_d = common_destination[0]

        time_o = _timetable_seconds(
            timetable[timetable["line_id"] == line_o],
            origin_reference,
            transfer_name,
        )
        time_d = _timetable_seconds(
            timetable[timetable["line_id"] == line_d],
            destination_reference,
            transfer_name,
        )

        enter_time = walking_times.loc[
            walking_times["station_name"] == origin_reference,
            "shortest_walking_time",
        ]
        exit_time = walking_times.loc[
            walking_times["station_name"] == destination_reference,
            "shortest_walking_time",
        ]
        if enter_time.empty or exit_time.empty:
            continue

        origin_id_rows = result.loc[
            result["station_name"] == origin_reference, "station_id"
        ]
        destination_id_rows = result.loc[
            result["station_name"] == destination_reference, "station_id"
        ]
        if origin_id_rows.empty or destination_id_rows.empty:
            continue

        origin_id = int(origin_id_rows.iloc[0])
        destination_id = int(destination_id_rows.iloc[0])

        matching = trips[
            (trips["origin_station"] == origin_id)
            & (trips["destination_station"] == destination_id)
        ].sort_values("duration")

        if matching.empty:
            continue

        shortest = float(matching.iloc[0]["duration"])
        transfer_time = (
            shortest
            - time_o
            - time_d
            - float(enter_time.iloc[0])
            - float(exit_time.iloc[0])
        )
        result.loc[row_idx, "transfer_time"] = transfer_time

    return result


def combine_station_times(stations, walking_times, waiting_times, transfer_times):
    out = stations.copy()

    out = out.drop(
        columns=[
            c
            for c in ["shortest_walking_time", "waiting_time", "transfer_time"]
            if c in out.columns
        ],
        errors="ignore",
    )

    out = out.merge(
        walking_times[["station_id", "shortest_walking_time"]],
        on="station_id",
        how="left",
    )
    out = out.merge(
        waiting_times[["station_id", "waiting_time"]],
        on="station_id",
        how="left",
    )
    out = out.merge(
        transfer_times[["station_id", "transfer_time"]],
        on="station_id",
        how="left",
    )

    for c in ["shortest_walking_time", "waiting_time", "transfer_time"]:
        out[c] = out[c].fillna(0.0)

    return out


def build_subway_network(
    stations,
    timetable,
    transfer_links,
    base_stations,
    circular_line_id=None,
    circular_line_period_seconds=None,
):
    """Create the weighted metro graph used for candidate path generation."""
    graph = nx.Graph()
    graph.add_nodes_from(stations["station_id"].astype(int).tolist())

    for line in stations["line_id"].dropna().unique():
        temp_timetable = timetable[timetable["line_id"] == line]
        if circular_line_id is not None and line == circular_line_id:
            temp_timetable = temp_timetable[temp_timetable["direction"] == 0]

        line_stations = stations[stations["line_id"] == line].reset_index(drop=True)

        for idx in range(len(line_stations) - 1):
            a = int(line_stations.iloc[idx]["station_id"])
            b = int(line_stations.iloc[idx + 1]["station_id"])
            a_name = line_stations.iloc[idx]["station_name"]
            b_name = line_stations.iloc[idx + 1]["station_name"]

            weight = _timetable_seconds(temp_timetable, a_name, b_name)

            if (
                circular_line_id is not None
                and line == circular_line_id
                and circular_line_period_seconds is not None
                and weight >= circular_line_period_seconds / 2
            ):
                weight = circular_line_period_seconds - weight

            graph.add_edge(a, b, weight=float(weight))

        if (
            circular_line_id is not None
            and line == circular_line_id
            and len(line_stations) > 1
        ):
            a = int(line_stations.iloc[-1]["station_id"])
            b = int(line_stations.iloc[0]["station_id"])
            a_name = line_stations.iloc[-1]["station_name"]
            b_name = line_stations.iloc[0]["station_name"]
            weight = _timetable_seconds(temp_timetable, a_name, b_name)
            if (
                circular_line_period_seconds is not None
                and weight >= circular_line_period_seconds / 2
            ):
                weight = circular_line_period_seconds - weight
            graph.add_edge(a, b, weight=float(weight))

    for _, row in transfer_links.iterrows():
        a = int(row["station_id"])
        b = int(row["transfer_station_id"])

        a_name = stations.loc[stations["station_id"] == a, "station_name"]
        b_name = stations.loc[stations["station_id"] == b, "station_name"]
        if a_name.empty or b_name.empty:
            continue

        a_name = a_name.iloc[0]
        b_name = b_name.iloc[0]

        a_lines = base_stations[
            base_stations["station_name"] == a_name
        ]["line_id"].unique()
        b_lines = base_stations[
            base_stations["station_name"] == b_name
        ]["line_id"].unique()

        common = [line for line in a_lines if line in b_lines]
        if not common:
            continue

        line = common[0]
        temp_timetable = timetable[timetable["line_id"] == line]
        weight = _timetable_seconds(temp_timetable, a_name, b_name)
        graph.add_edge(a, b, weight=float(weight))

    return graph


def graph_edges_dataframe(graph):
    rows = [
        {"origin": int(a), "destination": int(b), "weight": float(d["weight"])}
        for a, b, d in graph.edges(data=True)
    ]
    return pd.DataFrame(rows)


def select_candidate_destinations(trips, user_id, top_k=3):
    user_trips = trips[trips["user_id"] == user_id]
    counts = (
        user_trips.groupby("destination_station")
        .size()
        .reset_index(name="destination_volume")
        .sort_values("destination_volume", ascending=False)
        .reset_index(drop=True)
    )
    return counts.head(top_k)["destination_station"].astype(int).tolist()


def k_shortest_paths(graph, source, target, k, weight=None):
    return list(
        islice(
            nx.shortest_simple_paths(graph, source, target, weight=weight),
            k,
        )
    )


def _best_candidate_feature(trips, mode):
    if mode == "legacy_max_destination_id":
        # Faithfully reproduces:
        # data.groupby('GRANT_CARD_CODE')['D_Station'].max()
        return (
            trips.groupby("user_id")["destination_station"]
            .max()
            .reset_index(name="best_candidate")
        )

    if mode == "most_frequent":
        counts = (
            trips.groupby(["user_id", "destination_station"])
            .size()
            .reset_index(name="n")
        )
        idx = counts.groupby("user_id")["n"].idxmax()
        return counts.loc[
            idx, ["user_id", "destination_station"]
        ].rename(columns={"destination_station": "best_candidate"})

    raise ValueError(
        "best_candidate_mode must be 'legacy_max_destination_id' or 'most_frequent'."
    )


def prepare_candidate_paths(
    graph,
    trips,
    station_times,
    candidate_destinations=3,
    paths_per_destination=3,
    best_candidate_mode="legacy_max_destination_id",
):
    destination_volume = (
        trips.groupby("destination_station")
        .size()
        .reset_index(name="destination_volume")
    )
    best_candidate = _best_candidate_feature(trips, best_candidate_mode)

    records = []

    for trip_index, row in trips.reset_index(drop=True).iterrows():
        if trip_index % 10000 == 0:
            print(f"Preparing candidate paths: {trip_index}/{len(trips)}")

        user_id = row["user_id"]
        origin = int(row["origin_station"])
        true_destination = int(row["destination_station"])

        candidates = select_candidate_destinations(
            trips, user_id, top_k=candidate_destinations
        )

        for destination in candidates:
            if origin not in graph or destination not in graph:
                continue

            try:
                paths = k_shortest_paths(
                    graph,
                    origin,
                    int(destination),
                    paths_per_destination,
                    weight=None,  # preserves the source script's call
                )
            except nx.NetworkXNoPath:
                continue

            for path in paths:
                path_weight = sum(
                    graph[path[n]][path[n + 1]]["weight"]
                    for n in range(len(path) - 1)
                )

                transfer_weight = 0.0
                for node in path[:-1]:
                    match = station_times.loc[
                        station_times["station_id"] == node, "transfer_time"
                    ]
                    if not match.empty:
                        transfer_weight += float(match.iloc[0])

                walk_origin = station_times.loc[
                    station_times["station_id"] == path[0],
                    "shortest_walking_time",
                ]
                walk_destination = station_times.loc[
                    station_times["station_id"] == path[-1],
                    "shortest_walking_time",
                ]
                walking_weight = (
                    float(walk_origin.iloc[0]) if not walk_origin.empty else 0.0
                ) + (
                    float(walk_destination.iloc[0])
                    if not walk_destination.empty
                    else 0.0
                )

                records.append(
                    {
                        "trip_index": trip_index,
                        "user_id": user_id,
                        "origin_station": origin,
                        "path_weight": float(path_weight),
                        "transfer_weight": float(transfer_weight),
                        "walking_weight": float(walking_weight),
                        "destination_station": int(destination),
                        "true_destination": true_destination,
                        "path": str(path),
                    }
                )

    candidate_df = pd.DataFrame(records)
    if candidate_df.empty:
        raise ValueError("No candidate paths could be generated.")

    candidate_df = candidate_df.merge(
        best_candidate, on="user_id", how="left"
    )
    candidate_df = candidate_df.merge(
        destination_volume, on="destination_station", how="left"
    )

    cols = [
        "trip_index",
        "user_id",
        "origin_station",
        "path_weight",
        "transfer_weight",
        "walking_weight",
        "destination_volume",
        "best_candidate",
        "destination_station",
        "true_destination",
        "path",
    ]
    return candidate_df[cols]


def generate_trajectory_scores_mnlogit(candidate_df, destination_classes):
    """Generate trajectory score vectors using statsmodels MNLogit.

    This is a minimal replacement of the previous OLS fitting step. The
    candidate-path table and the downstream destination-probability aggregation
    are otherwise kept in the same form for the demonstration repository.

    Parameters are estimated by maximum likelihood through ``sm.MNLogit.fit``.
    """
    feature_cols = [
        "origin_station",
        "path_weight",
        "transfer_weight",
        "walking_weight",
        "destination_volume",
        "best_candidate",
        "destination_station",
    ]

    working = candidate_df.copy()

    # Encode destination-station labels as contiguous MNLogit classes.
    class_labels = sorted(
        working["true_destination"].astype(int).unique().tolist()
    )
    class_to_code = {
        label: code for code, label in enumerate(class_labels)
    }

    y = (
        working["true_destination"]
        .astype(int)
        .map(class_to_code)
        .astype(int)
    )

    # Standardize explanatory variables only for numerical stability.
    X = working[feature_cols].astype(float).copy()
    means = X.mean()
    stds = X.std(ddof=0).replace(0.0, 1.0)
    X_scaled = (X - means) / stds
    X_scaled = sm.add_constant(X_scaled, has_constant="add")

    model = sm.MNLogit(y, X_scaled)

    # MNLogit uses maximum-likelihood estimation.
    result = model.fit(
        method="bfgs",
        maxiter=200,
        disp=False,
    )

    # Probability of every true-destination class for every candidate-path row.
    probability_matrix = np.asarray(
        result.predict(X_scaled),
        dtype=float,
    )

    # For each path row, use the MNLogit probability associated with that
    # row's candidate destination. If a candidate destination was not observed
    # as a true-destination class, assign a very small score.
    candidate_scores = []

    for row_pos, destination in enumerate(
        working["destination_station"].astype(int).tolist()
    ):
        code = class_to_code.get(destination)

        if code is None or code >= probability_matrix.shape[1]:
            candidate_scores.append(1e-12)
        else:
            candidate_scores.append(
                float(probability_matrix[row_pos, code])
            )

    working["path_probability"] = candidate_scores

    # Preserve the previous workflow: normalize candidate-path scores within
    # each trip and aggregate them to destination-level trajectory scores.
    group_sum = working.groupby("trip_index")[
        "path_probability"
    ].transform("sum")

    working["path_probability"] = np.where(
        group_sum > 0,
        working["path_probability"] / group_sum,
        0.0,
    )

    score_rows = []
    meta_rows = []

    for trip_index in sorted(working["trip_index"].unique()):
        temp = working[
            working["trip_index"] == trip_index
        ].copy()

        destination_probability = (
            temp.groupby("destination_station")["path_probability"]
            .sum()
            .to_dict()
        )

        row = {
            f"dest_{int(destination)}": float(
                destination_probability.get(destination, 0.0)
            )
            for destination in destination_classes
        }
        score_rows.append(row)

        predicted_destination = max(
            destination_probability,
            key=destination_probability.get,
        )

        # Keep the existing trajectory-selection convention.
        chosen = (
            temp[
                temp["destination_station"]
                == predicted_destination
            ]
            .sort_values(
                "path_probability",
                ascending=True,
            )
            .iloc[0]
        )

        meta_rows.append(
            {
                "trip_index": int(trip_index),
                "predicted_destination": int(
                    predicted_destination
                ),
                "true_destination": int(
                    chosen["true_destination"]
                ),
                "selected_path": chosen["path"],
                "estimated_total_time": float(
                    chosen["path_weight"]
                    + chosen["transfer_weight"]
                    + chosen["walking_weight"]
                ),
            }
        )

    scores = pd.DataFrame(score_rows)
    metadata = pd.DataFrame(meta_rows)

    return scores, metadata, result

