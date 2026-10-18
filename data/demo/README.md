# Demo data files

All files in this folder use the same anonymized station numbering.

- `trips_features_demo.csv`: feature-engineered trip records.
- `stations_demo.csv`: anonymized station attributes.
- `base_stations_demo.csv`: station-line membership.
- `timetable_demo.csv`: representative anonymized timetable.
- `transfer_links_demo.csv`: transfer relationships.
- `topology_demo.csv`: bidirectional graph edges used by the demo model.

`trips_features_demo.csv` is intentionally header-only in this package because
the final feature-engineered trip file must be generated locally from the
restricted source data before public upload.
