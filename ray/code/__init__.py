"""BOXRR head-movement pipeline.

Run it with:  python main.py   (the menu, or see main.py for the download / export commands)
All data (cache/, saved/, h5/) goes into <project root>/data, see DATA_DIR in config.py.

Where things are (each file is short and does one job):

  config.py      constants and paths (change settings here)
  cli.py         ask_int / ask_float / human_size
  timefmt.py     seconds vs milliseconds vs Unix time detection
  head.py        find the head device / game from the metadata (skips Tilt Brush etc.)
  storage.py     folder layout: users/ (old extraction), saved/ (passes)
  state.py       cache/processed.json (finished tars)
  cleaning.py    the rules, shared by the filter and the export: head channel, cleaning,
                 motion filter (sweep S, velocity V), 100 Hz resample
  processing.py  the filter: process_func (runs cleaning.py, logs S and V)
  results_db.py  cache/results.sqlite: one row per recording, evaluate_replay
  hf.py          Hugging Face: tar list, reading replays out of a tar
  pipeline.py    options 1, 4, 8, 14: download tars, check recordings, keep passes,
                 download until every train / validation / test split can be filled
  extracted.py   options 2, 6: tars still extracted under users/
  reports.py     startup status, option 5, option 7
  viewer.py      option 3: step-by-step inspector
  maintenance.py options 9-12: re-check, restore, repair, purge
  h5_export.py   option 13: HDF5 export (one file, train / validation / test inside it)
  legacy.py      DEPRECATED: cache/manifest.json and the old extraction flow
"""
import numpy as np

np.set_printoptions(suppress=True, precision=4)
