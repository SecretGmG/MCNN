"""Debug trace for the 5x4x5 binary integration fixture.

Shows full inference: input bits, hidden layer binary activations,
output layer clamped to 0–15.

Usage: .venv/bin/python experiments/basic_task/debug_integration.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from mcnn import discrete_from_state_dict


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "integration"


def main():
    artifact = torch.load(ARTIFACTS / "mcnn_binary_5x4x5_integration.pt", map_location="cpu")
    model = discrete_from_state_dict(artifact["model"])
    inputs = artifact["task"]["inputs"].to(torch.float32)
    targets = artifact["task"]["targets"].to(torch.float32)

    lines = ["MCNN Debug Trace — 5x4x5 Binary Integration", "=" * 60]

    for example_idx in range(len(inputs)):
        inp = inputs[example_idx : example_idx + 1]
        with torch.no_grad():
            out = model.forward(inp)
        out_clamped = out.round().clamp(0, 15).squeeze().int()
        expected = targets[example_idx].round().clamp(0, 15).int()

        match = "✓" if torch.all(out_clamped == expected) else "✗"
        lines.append(f"\nExample {example_idx} {match}")
        lines.append("  Input: " + " ".join(str(int(b)) for b in inp.squeeze().int().tolist()))

        lines.append("  Output (clamped 0–15):")
        for i in range(out_clamped.shape[0]):
            lines.append(f"    y{i}: {out_clamped[i].item():2d} (want {expected[i].item():2d})")

        # Hidden layer activations
        hidden = inp.squeeze().int()
        for layer_idx, layer in enumerate(model.layers):
            w = layer.weight.to(dtype=hidden.dtype)
            b = layer.bias.to(dtype=hidden.dtype)
            pre = (w @ hidden + b).int()

            if layer.activation == "binary_step":
                active = (pre > 0).int()
                lines.append(f"  Layer {layer_idx} ({active.shape[0]} units, {active.sum().item()} active, binary_step):")
                for u in range(active.shape[0]):
                    if active[u].item():
                        lines.append(f"    h{u:02d}: pre={pre[u].item():+4d} → 1")
                hidden = active
            else:
                out_layer = pre.round().clamp(layer.output_min, layer.output_max).int()
                lines.append(f"  Layer {layer_idx} ({layer.weight.shape[0]} units, bounded_int [{int(layer.output_min)},{int(layer.output_max)}]):")
                for u in range(out_layer.shape[0]):
                    if out_layer[u].item() != 0:
                        lines.append(f"    u{u:02d}: pre={pre[u].item():+4d} → {out_layer[u].item()}")
                hidden = out_layer

    trace = "\n".join(lines)
    out_path = ARTIFACTS / "debug_trace.txt"
    out_path.write_text(trace)
    print(f"saved {out_path}")

    # Summary
    inp_t = inputs.clone()
    with torch.no_grad():
        all_out = model.forward(inp_t)
    all_pred = all_out.round().clamp(0, 15)
    matches = (all_pred == targets.round().clamp(0, 15)).all(dim=1)
    print(f"Row accuracy: {matches.float().mean().item():.3f} ({matches.sum().item()}/{len(matches)})")


if __name__ == "__main__":
    main()
