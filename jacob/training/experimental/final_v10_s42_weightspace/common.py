"""Fixed identities, atomic records and the approved user-level protocol."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[2]
REPO = TRAINING.parents[1]
ROOT = TRAINING / "outputs/final_v10_s42_weightspace_20261111"
OUTPUT = ROOT / "execution"
SOURCE = REPO / "jacob/models/tcnn_v9_k8_combined_moredata_final_v10_s42_stream.onnx"
ORIGINAL = ROOT / "originals/final_s42.onnx"
DATA = REPO / "data/abc_moredata_final/abc_moredata_final/moredata_final.h5"
MANIFEST = DATA.with_name("manifest_moredata_final.json")
OLD_MANIFEST = REPO / "data/h5/manifest_merged_20261101_140915.json"
MODEL_SHA = "28221e1e0b559e8d4d900f8b448acd22a0fbe0da38f904cb12d4e268aca53cad"
DATA_SHA = "431d7c72dc8e059bbaea76afa3b64a751d1cfcc563c58bddcd639e8361ffe00f"
MANIFEST_SHA = "f00497dc32317ad864e0041e3da764c4bd5adc1b92eafcedff318eec454a62dc"
PENALTY = 0.0001
STRENGTHS = (0.25, 0.5, 1.0)
MAX_SECONDS = 6 * 60 * 60
ANGLE_TOLERANCE = 0.001


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def validation_partition(validation, old_validation):
    """Keep all old development users in discovery; hash only the newer users."""
    users, old = set(validation), set(old_validation)
    if len(users) != 786 or len(old) != 120 or not old <= users:
        raise ValueError("Unexpected validation membership")
    newer = sorted(users - old, key=lambda u: (
        hashlib.sha256(f"final-v10-s42-weightspace-v1:{u}".encode()).hexdigest(), u))
    discovery = sorted(old | set(newer[:404]))
    assessment = sorted(newer[404:])
    assert len(discovery) == 524 and len(assessment) == 262
    assert not set(discovery) & set(assessment)
    return discovery, assessment


def window_count(rows, stride):
    return sum((row["samples"] - 420) // stride + 1 for row in rows)


def check_originals():
    for path in (SOURCE, ORIGINAL, ROOT / "controls/unchanged.onnx", ROOT / "working/final_s42.onnx"):
        if sha256(path) != MODEL_SHA:
            raise ValueError(f"Original preservation failed: {path}")


def assessment_gate(summaries, comparisons):
    original, candidate = summaries["identity"], summaries["candidate"]
    paired = comparisons["identity"]
    checks = {
        "assessment_users_262": paired["users"] == 262,
        "improvement_at_least_0_005_deg": paired["mean_diff_deg"] <= -0.005,
        "ci95_below_zero": paired["ci95"][1] < 0,
        "equal_user_mean_lower": paired["equal_user_mean_diff_deg"] < 0,
        "majority_users_improved": paired["users_better"] > paired["users"] / 2,
        "p95_no_worse": candidate["p95_endpoint_deg"] <= original["p95_endpoint_deg"],
        "all_horizons_mean_no_worse": candidate["mean_all_horizons_deg"] <= original["mean_all_horizons_deg"],
        "beats_gain_control": comparisons["global_105"]["mean_diff_deg"] < 0,
    }
    return {"passed": all(checks.values()), "checks": checks}
