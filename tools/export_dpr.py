"""Export the DPR 512 Hourglass relighting net to ONNX for onnxruntime-web.

Input  l : (1,1,512,512) float32, Lab L-channel in [0,1]
Input  sh: (1,9,1,1)    float32, order-2 spherical-harmonic lighting
Output out: (1,1,512,512) float32, relit L in [0,1]
skip_count is fixed to 0 (as in the reference demo) and baked into the graph.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "DPR", "model"))

from defineHourglass_512_gray_skip import HourglassNet  # noqa: E402

WEIGHTS = os.path.join(HERE, "DPR", "trained_model", "trained_model_03.t7")
OUT = os.path.join(HERE, "dpr_512.onnx")


class Wrap(torch.nn.Module):
    """Fix skip_count=0 and drop the predicted-lighting output."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, l, sh):
        out_img, _ = self.net(l, sh, 0)
        return out_img


def main():
    net = HourglassNet()
    state = torch.load(WEIGHTS, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    model = Wrap(net).eval()

    dummy_l = torch.zeros(1, 1, 512, 512, dtype=torch.float32)
    dummy_sh = torch.zeros(1, 9, 1, 1, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_l, dummy_sh),
            OUT,
            input_names=["l", "sh"],
            output_names=["out"],
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"exported {OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
