import sys
sys.path.insert(0, r"<repo>\jacob\training")
import onnx
from experimental.v7_compact.model import load_checkpoint
from experimental.v7_compact.export import export_streaming_onnx
model = load_checkpoint(r"<repo>\jacob\training\outputs\v8_recipe_sep_base3_addons\angular_loss\seed_44\best.pt")[0]
path, shapes = export_streaming_onnx(model, sys.argv[1])
g = onnx.load(str(path)).graph
nodes = [n for n in g.node if n.op_type != "Constant"]
print(len(nodes), "non-Constant nodes;", len(g.node), "total")
for i, n in enumerate(nodes):
    print(i, n.op_type, list(n.input)[:3], "->", list(n.output))
