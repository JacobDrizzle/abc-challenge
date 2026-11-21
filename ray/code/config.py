"""Constants and paths shared by every module. Change settings here."""

import os


REPO = "cschell/boxrr-23"


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
# <project root>/data (gitignored). main.py changes into it at start, so the relative paths below
# (cache/, saved/, h5/ and the old users/ extraction folder) all end up inside it.


CACHED_DIR = "cache"


TAR_LIST_PATH = os.path.join(CACHED_DIR, "tars.json")      # [tar_name, ...]


TAR_INDEX = 40090


SAVED_ROOT = "saved"


EXCLUDED_GAMES = {"tilt brush"}  # lower-case game names to ignore: they have no head tracking


RESULTS_DB = os.path.join(CACHED_DIR, "results.sqlite")  # every recording, pass or fail


TMP_CACHE_DIR = os.path.join(CACHED_DIR, "hf_tmp")      # tars are downloaded here, then deleted


PROCESSED_PATH = os.path.join(CACHED_DIR, "processed.json")  # {tar_name: {total, passed, errors}}


DOWNLOAD_MAX_ERRORS_IN_ROW = 10  # download_users stops after this many failed tars in a row (Hugging Face down or rate limiting)


PREPARATION_VERSION = "group-a-v2"  # stamped on every output. Bump it whenever a rule below changes


# The challenge filter (Data page steps 1-8), the same as Group B's reference (abc_data/config.py).
# cleaning.py applies these rules for both process_func (what goes into saved/) and the HDF5 export.


TRIM_S = 5.0                      # step 1: drop the first 5 s of accumulated positive time steps


WINDOW_S = 5.0                    # step 2: non-overlapping 5 s windows, also the velocity denominator


MIN_WINDOW_SAMPLES = 10           # step 2: windows with fewer samples are skipped


MIN_ASSESSED_SAMPLES = 20         # step 2: sessions with fewer samples after the trim are not assessed


EULER_ORDER = "xyz"               # step 3: lower case = extrinsic in scipy


SWEEP_PERCENTILES = (10, 90)      # step 4: sweep = P90 - P10, on the angles as computed (no unwrap)


SWEEP_MIN_DEG = 17.0              # step 7: strict >


VELOCITY_MIN_DEG_S = 15.0         # step 7: strict >


RATE_BAND_HZ = (85.0, 110.0)      # step 8: inclusive, from the median positive time step


MIN_QUAT_NORM = 1e-8              # quaternions shorter than this are invalid samples


H5_DIR = "h5"


H5_CACHE_PATH = os.path.join(CACHED_DIR, "h5_exported.json")  # which users were already written, and where


H5_SPLITS: tuple[str, ...] = ("train", "validation", "test")  # the group names Group B's loader reads


H5_DEFAULT_USERS = (100, 15, 20)  # training, validation, test. Each "user" is one tar file


H5_RATE_HZ = 100                  # every session is put on a uniform 100 Hz grid


H5_MIN_SAMPLES = 1000             # shorter segments are dropped (the hard floor is 420 = 400 context + 20 targets)


H5_MAX_GAP_S = 0.10               # gaps up to this are SLERP-filled, longer ones split the session


H5_MAX_RECORDINGS_PER_USER = 5    # qualified recordings kept per user, picked in a seeded order


H5_COMPRESSION = "gzip"           # None = bigger files but faster reads


H5_SEED = 42                      # seed for shuffling users into splits and ordering each user's recordings


RESERVED_SPLIT_PATH = os.environ.get("ABC_RESERVED_SPLIT",
                                     os.path.join(DATA_DIR, "reserved", "frozen_user_split.json"))
# Group B's frozen split (not in the repo). Its test_users and retired_test_users never go into train or validation.
# Set ABC_RESERVED_SPLIT to read it from elsewhere. Without one, a file holding {"test_users": []} reserves nobody.
