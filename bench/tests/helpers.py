"""Tiny synthetic ONNX graphs and HDF5 files for the unit tests (no torch, no real data)."""
import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bench.geometry import random_motion

FLOAT = TensorProto.FLOAT


def save(graph, path, opset=17):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def const(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def _hold_last_nodes(context, prediction, scale):
    """prediction = scale * repeat(context[:, -1:], 20): hold-last with an unnormalised output."""
    nodes = [helper.make_node("Slice", [context, "s_start", "s_end", "s_axis"], ["last"]),
             helper.make_node("Tile", ["last", "repeats"], ["tiled"]),
             helper.make_node("Mul", ["tiled", "scale"], [prediction])]
    inits = [const("s_start", np.array([-1], np.int64)), const("s_end", np.array([400], np.int64)),
             const("s_axis", np.array([1], np.int64)), const("repeats", np.array([1, 20, 1], np.int64)),
             const("scale", np.array(scale, np.float32))]
    return nodes, inits


def hold_last_graph(path, context="foo", prediction="bar", scale=2.0):
    nodes, inits = _hold_last_nodes(context, prediction, scale)
    graph = helper.make_graph(nodes, "hold_last", [helper.make_tensor_value_info(context, FLOAT, [1, 400, 4])],
                              [helper.make_tensor_value_info(prediction, FLOAT, [1, 20, 4])], inits)
    return save(graph, path)


def counter_graph(path):
    """Stateful hold-last: state_out_0 = state_in_0 + 1, state_out_1 = state_in_1 + 2. Outputs deliberately
    listed out of order, so feeding states back by position instead of by name would fail."""
    nodes, inits = _hold_last_nodes("context", "prediction", 1.0)
    nodes += [helper.make_node("Add", ["state_in_0", "one"], ["state_out_0"]),
              helper.make_node("Add", ["state_in_1", "two"], ["state_out_1"])]
    inits += [const("one", np.array(1, np.float32)), const("two", np.array(2, np.float32))]
    graph = helper.make_graph(
        nodes, "counter",
        [helper.make_tensor_value_info("context", FLOAT, [1, 400, 4]),
         helper.make_tensor_value_info("state_in_0", FLOAT, [1, 3]),
         helper.make_tensor_value_info("state_in_1", FLOAT, [2, 2])],
        [helper.make_tensor_value_info("state_out_1", FLOAT, [2, 2]),
         helper.make_tensor_value_info("prediction", FLOAT, [1, 20, 4]),
         helper.make_tensor_value_info("state_out_0", FLOAT, [1, 3])], inits)
    return save(graph, path)


def make_h5(path, layout):
    """layout: {user: [(recording, samples, qualified), ...]}; one segment per entry, smooth motion."""
    seed = 0
    with h5py.File(path, "w") as h5:
        h5.attrs["quaternion_order"] = "wxyz"
        h5.attrs["rate_hz"] = 100
        for user, entries in layout.items():
            for i, (recording, samples, qualified) in enumerate(entries):
                seed += 1
                group = h5.create_group(f"{user}/{recording}/segment_{i:03d}")
                group["quaternion_wxyz"] = random_motion(samples, seed)
                group["time_s"] = np.arange(samples) / 100.0
                group.attrs["challenge_filter_pass"] = qualified
    return path


def nest_in_splits(flat_path, path, members):
    """The flat file's players regrouped under split groups, as the ray/ export writes them.
    members: {split: [user, ...]}."""
    with h5py.File(flat_path, "r") as src, h5py.File(path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs.create("splits", list(members), dtype=h5py.string_dtype())
        for split, users in members.items():
            dst.create_group(split)
            for user in users:
                src.copy(src[user], dst[split], name=user)
    return path


def definition(players, name="synthetic"):
    return {"name": name, "players": sorted(players),
            "protocol": {"context": 400, "horizon": 20, "rate_hz": 100.0, "stride": 1}}
