# Tracking fixtures — provenance

`metrica_sample2_home.csv` / `metrica_sample2_away.csv`:

- Source: Metrica Sports open sample data, `metrica-sports/sample-data`
  repository (MIT-licensed), Sample Game 2 raw tracking files.
- Truncation: the first 2,002 lines (2 header rows + 2,000 frames ≈ 80 s
  of play at 25 fps) of each ~30 MB file, so a genuine vendor file can be
  committed without bloating the repository. The rows are unmodified
  vendor data, not synthetic.
- Layout: two header rows (team row, jersey row), then
  `Period, Frame, Time [s], PlayerN_x, PlayerN_y, ..., Ball_x, Ball_y`
  in normalized [0,1] pitch coordinates; the Metrica loader converts to
  Kawkab meters on import.
