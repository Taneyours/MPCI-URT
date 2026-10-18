"""
Prepare a consistent, anonymized set of demo files for the MPCI-URT GitHub repository.

INPUT
-----
1. A previously filtered feature-engineered trip file:
   物理信息实验数据_1号线2号线_demo.csv

   The following six station-related fields must already all belong to
   Line 1 / Line 2:
       O_Station
       D_Station
       user_best_O
       user_best_D
       user_second_D
       O_best_D

2. station_index_id_50站点.csv
3. msim_base_stations_2022.csv
4. msim_t_timetables.csv
5. 换乘站点连图信息.csv

OUTPUT
------
A public folder containing only anonymized / model-ready files:

github_demo_data/
├── trips_features_demo.csv
├── stations_demo.csv
├── base_stations_demo.csv
├── timetable_demo.csv
├── transfer_links_demo.csv
└── topology_demo.csv

IMPORTANT
---------
- Raw card IDs are never exported.
- Raw station names are never exported.
- Raw AFC station access codes are never exported.
- Exact AFC entry/exit timestamps are not exported.
- User IDs, station IDs, station names and line IDs are made consistent across
  all output files.
- This dataset is only intended to demonstrate the workflow, not to reproduce
  the paper's reported numerical results.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# 1. PATHS -- MODIFY THESE
# =====================================================================

TRIPS_FILE = Path(
    r"F:\YOUR_PATH\物理信息实验数据_1号线2号线_demo.csv"
)

STATION_INDEX_FILE = Path(
    r"F:\YOUR_PATH\station_index_id_50站点.csv"
)

BASE_STATIONS_FILE = Path(
    r"F:\YOUR_PATH\msim_base_stations_2022.csv"
)

TIMETABLE_FILE = Path(
    r"F:\YOUR_PATH\msim_t_timetables.csv"
)

TRANSFER_LINKS_FILE = Path(
    r"F:\YOUR_PATH\换乘站点连图信息.csv"
)

OUTPUT_DIR = Path(
    r"F:\YOUR_PATH\MPCI-URT\data\demo"
)


# =====================================================================
# 2. SELECTED LINES
# =====================================================================

# Internal line codes in the supplied data.
SELECTED_LINES = ("L010", "L020")

# Public/demo line labels.
PUBLIC_LINE_MAP = {
    "L010": "L001",
    "L020": "L002",
}


# =====================================================================
# 3. HELPERS
# =====================================================================

def read_csv_with_fallback(
    path,
    encodings=("utf-8", "gbk", "latin1"),
    **kwargs,
):
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                **kwargs,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def build_public_station_mapping(station_index):
    """
    Use station_index_id_50站点.csv as the authoritative station universe.

    Only stations whose BlongedLineNumber is L010 or L020 are retained.
    Public station IDs are consecutive integers 0, 1, ..., N-1.
    Public station names are S000, S001, ...
    """
    selected = station_index[
        station_index["BlongedLineNumber"].isin(
            SELECTED_LINES
        )
    ].copy()

    selected["index"] = pd.to_numeric(
        selected["index"],
        errors="raise",
    ).astype(int)

    selected = (
        selected
        .sort_values("index")
        .drop_duplicates("index")
        .reset_index(drop=True)
    )

    original_id_to_public_id = {
        int(original_id): int(public_id)
        for public_id, original_id
        in enumerate(selected["index"].tolist())
    }

    original_name_to_public_id = {
        row["StationName"]:
            original_id_to_public_id[int(row["index"])]
        for _, row in selected.iterrows()
    }

    original_name_to_public_name = {
        name: f"S{public_id:03d}"
        for name, public_id
        in original_name_to_public_id.items()
    }

    return (
        selected,
        original_id_to_public_id,
        original_name_to_public_id,
        original_name_to_public_name,
    )


# =====================================================================
# 4. PREPARE TRIP FEATURE FILE
# =====================================================================

def prepare_trip_features(
    trips,
    original_id_to_public_id,
):
    """
    Anonymize user IDs and remap every station-valued model feature using
    exactly the same station mapping.
    """

    required_columns = [
        "GRANT_CARD_CODE",
        "O_Station",
        "D_Station",
        "user_best_O",
        "user_best_D",
        "user_second_D",
        "O_best_D",
        "Duration",
        "hour",
        "week",
    ]

    missing = [
        col
        for col in required_columns
        if col not in trips.columns
    ]

    if missing:
        raise ValueError(
            "Trip file is missing required columns: "
            + ", ".join(missing)
        )

    df = trips.copy()

    station_columns = [
        "O_Station",
        "D_Station",
        "user_best_O",
        "user_best_D",
        "user_second_D",
        "O_best_D",
    ]

    # Standardize station-valued columns.
    for col in station_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        ).astype(int)

        unknown_values = (
            set(df[col].unique())
            - set(original_id_to_public_id)
        )

        if unknown_values:
            raise ValueError(
                f"{col} contains station IDs outside "
                f"the selected Line 1/Line 2 set: "
                f"{sorted(unknown_values)}"
            )

        df[col] = (
            df[col]
            .map(original_id_to_public_id)
            .astype(int)
        )

    # --------------------------------------------------------------
    # User anonymization
    # --------------------------------------------------------------
    #
    # The mapping exists only in memory.
    # No GRANT_CARD_CODE <-> user_id table is written to disk.
    #
    public_user_ids, _ = pd.factorize(
        df["GRANT_CARD_CODE"].astype(str),
        sort=False,
    )

    df["user_id"] = public_user_ids.astype(int)

    # --------------------------------------------------------------
    # Only retain already engineered/model-ready information.
    #
    # ENTRY_TIME and DEAL_TIME are intentionally excluded.
    # --------------------------------------------------------------
    output = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(int),
            "origin_station":
                df["O_Station"].astype(int),
            "destination_station":
                df["D_Station"].astype(int),
            "duration":
                pd.to_numeric(
                    df["Duration"],
                    errors="raise",
                ).round().astype(int),
            "hour":
                pd.to_numeric(
                    df["hour"],
                    errors="raise",
                ).astype(int),
            "weekday":
                pd.to_numeric(
                    df["week"],
                    errors="raise",
                ).astype(int),
            "user_best_origin":
                df["user_best_O"].astype(int),
            "user_best_destination":
                df["user_best_D"].astype(int),
            "user_second_destination":
                df["user_second_D"].astype(int),
            "origin_best_destination":
                df["O_best_D"].astype(int),
        }
    )

    return output


# =====================================================================
# 5. PREPARE STATION FILE
# =====================================================================

def prepare_stations(
    selected_station_index,
    original_name_to_public_id,
    original_name_to_public_name,
    base_stations,
):
    """
    One row per public physical station.

    line_id is a primary line label used by the lightweight trajectory code.
    Full station-line membership is kept separately in base_stations_demo.csv.
    """

    base_selected = base_stations[
        base_stations["BlongedLineNumber"]
        .isin(SELECTED_LINES)
        & base_stations["StationName"]
        .isin(original_name_to_public_id)
    ].copy()

    membership_count = (
        base_selected
        .groupby("StationName")["BlongedLineNumber"]
        .nunique()
        .to_dict()
    )

    rows = []

    for _, row in selected_station_index.iterrows():
        original_name = row["StationName"]
        original_line = row["BlongedLineNumber"]

        transfer_origin_name = ""
        transfer_destination_name = ""

        raw_o = row.get("O_Transfer", np.nan)
        raw_d = row.get("D_Transfer", np.nan)

        if (
            pd.notna(raw_o)
            and raw_o in original_name_to_public_name
        ):
            transfer_origin_name = (
                original_name_to_public_name[raw_o]
            )

        if (
            pd.notna(raw_d)
            and raw_d in original_name_to_public_name
        ):
            transfer_destination_name = (
                original_name_to_public_name[raw_d]
            )

        rows.append(
            {
                "station_id": int(
                    original_name_to_public_id[
                        original_name
                    ]
                ),
                "station_name":
                    original_name_to_public_name[
                        original_name
                    ],
                "line_id":
                    PUBLIC_LINE_MAP[original_line],
                "station_type": (
                    "transfer"
                    if membership_count.get(
                        original_name, 0
                    ) > 1
                    or row["StationType"] == "换乘站"
                    else "regular"
                ),
                "transfer_origin_name":
                    transfer_origin_name,
                "transfer_destination_name":
                    transfer_destination_name,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "station_id"
    ).reset_index(drop=True)


# =====================================================================
# 6. PREPARE BASE-STATION / LINE MEMBERSHIP FILE
# =====================================================================

def prepare_base_stations(
    base_stations,
    original_name_to_public_id,
    original_name_to_public_name,
):
    df = base_stations[
        base_stations["BlongedLineNumber"]
        .isin(SELECTED_LINES)
        & base_stations["StationName"]
        .isin(original_name_to_public_id)
    ].copy()

    df = df[
        [
            "StationName",
            "BlongedLineNumber",
        ]
    ].drop_duplicates()

    df["station_id"] = (
        df["StationName"]
        .map(original_name_to_public_id)
        .astype(int)
    )

    df["station_name"] = (
        df["StationName"]
        .map(original_name_to_public_name)
    )

    df["line_id"] = (
        df["BlongedLineNumber"]
        .map(PUBLIC_LINE_MAP)
    )

    return (
        df[
            [
                "station_id",
                "station_name",
                "line_id",
            ]
        ]
        .sort_values(
            ["line_id", "station_id"]
        )
        .reset_index(drop=True)
    )


# =====================================================================
# 7. PREPARE REPRESENTATIVE TIMETABLE
# =====================================================================

def prepare_timetable(
    timetable,
    original_name_to_public_id,
    original_name_to_public_name,
):
    """
    Keep only ONE representative direction-0 service for each line.

    The demo does not need a full operating-day timetable; it only needs a
    consistent example sufficient for the trajectory workflow to run.
    """

    parts = []

    selected_names = set(
        original_name_to_public_id
    )

    for original_line in SELECTED_LINES:
        line_df = timetable[
            (timetable["lineID"] == original_line)
            & (timetable["direction"] == 0)
            & (
                timetable["stationName"]
                .isin(selected_names)
            )
        ].copy()

        if line_df.empty:
            raise ValueError(
                f"No timetable rows found for "
                f"{original_line}"
            )

        line_df = (
            line_df
            .reset_index(drop=False)
            .rename(
                columns={
                    "index": "_source_row"
                }
            )
        )

        # Detect train-service blocks from station-order restarts.
        min_order = line_df["stationorder"].min()

        line_df["_service_id"] = (
            line_df["stationorder"]
            .eq(min_order)
            .cumsum()
        )

        coverage = (
            line_df
            .groupby("_service_id")[
                "stationName"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        service_id = coverage.index[0]

        service = line_df[
            line_df["_service_id"]
            == service_id
        ].copy()

        service = (
            service
            .sort_values(
                [
                    "stationorder",
                    "_source_row",
                ]
            )
            .drop_duplicates(
                "stationName",
                keep="first",
            )
        )

        service["arrivetime"] = (
            pd.to_datetime(
                service["arrivetime"],
                errors="raise",
            )
        )

        service["departuretime"] = (
            pd.to_datetime(
                service["departuretime"],
                errors="raise",
            )
        )

        # Shift real timetable dates to one artificial date.
        reference_time = min(
            service["arrivetime"].min(),
            service["departuretime"].min(),
        )

        artificial_start = pd.Timestamp(
            "2000-01-03 06:00:00"
        )

        service["arrival_time"] = (
            artificial_start
            + (
                service["arrivetime"]
                - reference_time
            )
        )

        service["departure_time"] = (
            artificial_start
            + (
                service["departuretime"]
                - reference_time
            )
        )

        service["station_id"] = (
            service["stationName"]
            .map(
                original_name_to_public_id
            )
            .astype(int)
        )

        service["station_name"] = (
            service["stationName"]
            .map(
                original_name_to_public_name
            )
        )

        service["line_id"] = (
            PUBLIC_LINE_MAP[
                original_line
            ]
        )

        part = service[
            [
                "line_id",
                "station_id",
                "station_name",
                "stationorder",
                "arrival_time",
                "departure_time",
                "direction",
            ]
        ].rename(
            columns={
                "stationorder":
                    "station_order"
            }
        )

        parts.append(part)

    return pd.concat(
        parts,
        ignore_index=True,
    )


# =====================================================================
# 8. PREPARE TRANSFER LINKS
# =====================================================================

def prepare_transfer_links(
    transfer_links,
    original_name_to_public_id,
):
    """
    Map by station NAME rather than by the original Station_index columns,
    because those index columns use a different full-network numbering system.
    """

    selected_names = set(
        original_name_to_public_id
    )

    df = transfer_links[
        transfer_links["StationName"]
        .isin(selected_names)
        & transfer_links["TransferName"]
        .isin(selected_names)
    ].copy()

    df["station_id"] = (
        df["StationName"]
        .map(original_name_to_public_id)
    )

    df["transfer_station_id"] = (
        df["TransferName"]
        .map(original_name_to_public_id)
    )

    return (
        df[
            [
                "station_id",
                "transfer_station_id",
            ]
        ]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .reset_index(drop=True)
    )


# =====================================================================
# 9. PREPARE SIMPLE TOPOLOGY
# =====================================================================

def prepare_topology(
    timetable_demo,
):
    """
    Build bidirectional edges from consecutive stations in each selected line.

    For public Line 2 (the circular line), connect the final representative
    station back to the first representative station.
    """

    edges = []

    for line_id, group in (
        timetable_demo
        .sort_values(
            [
                "line_id",
                "station_order",
            ]
        )
        .groupby("line_id")
    ):
        station_ids = (
            group["station_id"]
            .astype(int)
            .tolist()
        )

        for a, b in zip(
            station_ids[:-1],
            station_ids[1:],
        ):
            edges.append(
                {
                    "origin": a,
                    "destination": b,
                    "line_id": line_id,
                }
            )

            edges.append(
                {
                    "origin": b,
                    "destination": a,
                    "line_id": line_id,
                }
            )

        # Public L002 corresponds to the selected circular line L020.
        if (
            line_id == "L002"
            and len(station_ids) > 1
        ):
            a = station_ids[-1]
            b = station_ids[0]

            edges.append(
                {
                    "origin": a,
                    "destination": b,
                    "line_id": line_id,
                }
            )

            edges.append(
                {
                    "origin": b,
                    "destination": a,
                    "line_id": line_id,
                }
            )

    return (
        pd.DataFrame(edges)
        .drop_duplicates()
        .reset_index(drop=True)
    )


# =====================================================================
# 10. VALIDATION
# =====================================================================

def validate_consistency(
    trips_demo,
    stations_demo,
    base_demo,
    timetable_demo,
    transfer_demo,
    topology_demo,
):
    station_ids = set(
        stations_demo["station_id"]
        .astype(int)
    )

    trip_station_columns = [
        "origin_station",
        "destination_station",
        "user_best_origin",
        "user_best_destination",
        "user_second_destination",
        "origin_best_destination",
    ]

    for col in trip_station_columns:
        invalid = (
            set(
                trips_demo[col]
                .astype(int)
                .unique()
            )
            - station_ids
        )

        if invalid:
            raise ValueError(
                f"{col} contains invalid "
                f"public station IDs: "
                f"{sorted(invalid)}"
            )

    for name, df, cols in [
        (
            "base_stations_demo",
            base_demo,
            ["station_id"],
        ),
        (
            "timetable_demo",
            timetable_demo,
            ["station_id"],
        ),
        (
            "transfer_links_demo",
            transfer_demo,
            [
                "station_id",
                "transfer_station_id",
            ],
        ),
        (
            "topology_demo",
            topology_demo,
            [
                "origin",
                "destination",
            ],
        ),
    ]:
        for col in cols:
            invalid = (
                set(
                    df[col]
                    .astype(int)
                    .unique()
                )
                - station_ids
            )

            if invalid:
                raise ValueError(
                    f"{name}.{col} "
                    f"contains invalid station IDs: "
                    f"{sorted(invalid)}"
                )

    # User IDs should be compact 0..N-1.
    unique_users = sorted(
        trips_demo["user_id"]
        .astype(int)
        .unique()
    )

    expected_users = list(
        range(len(unique_users))
    )

    if unique_users != expected_users:
        raise ValueError(
            "user_id is not a compact "
            "0..N-1 sequence."
        )

    print(
        "Consistency check passed."
    )


# =====================================================================
# 11. MAIN
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("1. Reading source files")
    print("=" * 72)

    trips = read_csv_with_fallback(
        TRIPS_FILE
    )

    station_index = read_csv_with_fallback(
        STATION_INDEX_FILE,
        encodings=(
            "gbk",
            "utf-8",
            "latin1",
        ),
    )

    base_stations = read_csv_with_fallback(
        BASE_STATIONS_FILE,
        encodings=(
            "gbk",
            "utf-8",
            "latin1",
        ),
    )

    timetable = read_csv_with_fallback(
        TIMETABLE_FILE,
        encodings=(
            "utf-8",
            "gbk",
            "latin1",
        ),
    )

    transfer_links = read_csv_with_fallback(
        TRANSFER_LINKS_FILE
    )

    print("=" * 72)
    print("2. Building one shared anonymization mapping")
    print("=" * 72)

    (
        selected_station_index,
        original_id_to_public_id,
        original_name_to_public_id,
        original_name_to_public_name,
    ) = build_public_station_mapping(
        station_index
    )

    print(
        f"Public station count: "
        f"{len(selected_station_index)}"
    )

    print("=" * 72)
    print("3. Preparing trips_features_demo.csv")
    print("=" * 72)

    trips_demo = prepare_trip_features(
        trips,
        original_id_to_public_id,
    )

    print(
        f"Trip rows: "
        f"{len(trips_demo):,}"
    )

    print(
        f"Anonymized users: "
        f"{trips_demo['user_id'].nunique():,}"
    )

    print("=" * 72)
    print("4. Preparing auxiliary files")
    print("=" * 72)

    stations_demo = prepare_stations(
        selected_station_index,
        original_name_to_public_id,
        original_name_to_public_name,
        base_stations,
    )

    base_demo = prepare_base_stations(
        base_stations,
        original_name_to_public_id,
        original_name_to_public_name,
    )

    timetable_demo = prepare_timetable(
        timetable,
        original_name_to_public_id,
        original_name_to_public_name,
    )

    transfer_demo = prepare_transfer_links(
        transfer_links,
        original_name_to_public_id,
    )

    topology_demo = prepare_topology(
        timetable_demo
    )

    print("=" * 72)
    print("5. Checking cross-file consistency")
    print("=" * 72)

    validate_consistency(
        trips_demo,
        stations_demo,
        base_demo,
        timetable_demo,
        transfer_demo,
        topology_demo,
    )

    print("=" * 72)
    print("6. Saving public GitHub demo files")
    print("=" * 72)

    trips_demo.to_csv(
        OUTPUT_DIR
        / "trips_features_demo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stations_demo.to_csv(
        OUTPUT_DIR
        / "stations_demo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_demo.to_csv(
        OUTPUT_DIR
        / "base_stations_demo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    timetable_demo.to_csv(
        OUTPUT_DIR
        / "timetable_demo.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    transfer_demo.to_csv(
        OUTPUT_DIR
        / "transfer_links_demo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topology_demo.to_csv(
        OUTPUT_DIR
        / "topology_demo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFiles ready for GitHub:")

    for filename in [
        "trips_features_demo.csv",
        "stations_demo.csv",
        "base_stations_demo.csv",
        "timetable_demo.csv",
        "transfer_links_demo.csv",
        "topology_demo.csv",
    ]:
        print(
            "  ",
            OUTPUT_DIR / filename,
        )

    print(
        "\nNo card-ID mapping or "
        "original station-name mapping "
        "is written to disk."
    )


if __name__ == "__main__":
    main()
