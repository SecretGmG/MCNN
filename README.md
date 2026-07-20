# MCNN

MCNN is an experimental neural-network format intended to run quickly inside
Minecraft. The goal is to train with familiar ML tooling, then export a small
discrete network whose operations map cleanly to Minecraft logic.

## Core Network Rules

The Minecraft-side architecture is intentionally constrained:

- Output activation: bounded integer value.
- Output default range: integer values from `0` to `15`, inclusive.
- Effective weight default range: integer values from `-14` to `14`, inclusive.
- Bias: any positive value.
- Inference should use only the quantized/discrete form, not floating-point
  training parameters.

A neuron can be treated conceptually as:

```text
y = clamp_int(sum_i(w_i * x_i) + b, output_min, output_max)
```

Where:

- `x_i` is the previous layer output.
- `w_i` is an integer in the configured weight range.
- `b > 0`.
- `clamp_int(z, low, high)` rounds to an integer and clamps to the configured
  output range.

The default output convention is:

```text
output_min = 0
output_max = 15
```

The bounds should remain explicit constants in code and experiment configs so
the Minecraft runtime, PyTorch training path, exact inference, and exporter all
agree.

## Minecraft Layer Architecture

A layer is implemented as a chain of comparators subtracting signal strength
from a large initial signal, the bias. The bias signal is allowed to be larger
than vanilla redstone's normal `0..15` signal range. Bias sources are encoded as
barrels with glitched stacks of non-stackable items, currently capped to stack
counts of `64`, which represents every integer signal from `0` through `897`.
Each input connection in a neuron is implemented with one of two comparator
modes:

- Mode A: subtracts `15` by default, and subtracts a configured value `v` in
  `1..14` when the input is activated.
- Mode B: subtracts a configured value `v` in `1..14` by default, and subtracts
  `15` when the input is activated.

The useful contribution of a connection is the difference between the inactive
subtraction and the active subtraction:

```text
effective_weight = inactive_subtraction - active_subtraction
```

That gives the two mode encodings:

```text
Mode A: inactive = 15, active = v   -> effective_weight = 15 - v
Mode B: inactive = v,  active = 15  -> effective_weight = v - 15
```

The connection furnace value must not be `0`, because a zero-signal furnace
behaves like a normal block in this circuit. A connection can therefore
represent any integer effective weight from `-14` to `14`:

- Positive weights use Mode A with `v = 15 - weight`.
- Negative weights use Mode B with `v = weight + 15`.
- Zero weights use Mode C: omit the entire connection circuit.

Mode C reduces both physical model size and the required bias signal because the
connection contributes no default subtraction to the comparator chain.

Conceptually, the neuron computes the bias minus the default subtraction chain,
then each activated input adds its effective weight relative to that default.
The trained model can therefore be written and tested as:

```text
y = clamp_int(sum_i(w_i * x_i) + b, 0, output_max)
```

Where each nonzero `w_i` is an effective weight encoded by a comparator mode and
a `1..14` configured subtraction value.

## Training Direction

The practical training loop should separate three representations:

1. Trainable PyTorch parameters.
2. Quantized Minecraft-compatible parameters.
3. Exported Minecraft/runtime representation.

During early experiments, train with PyTorch using a straight-through estimator
so gradients can pass through the non-differentiable output and weight
quantization.

Recommended approach:

- Keep latent floating-point weights during training.
- Quantize weights in the forward pass to the configured integer range.
- Use a straight-through estimator for quantization.
- Use a straight-through estimator for the bounded integer output activation.
- Clamp or parameterize bias so it remains positive.
- Evaluate both the STE forward pass and the exact quantized inference path.

After quantization, optionally refine the discrete network with an evolutionary
method:

- Mutate integer weights directly.
- Mutate biases within a positive range.
- Score candidates on the task objective.
- Keep the best candidates with elitism.
- Use PyTorch training output as the seed population instead of starting from
  random networks.

This gives fast gradient-based iteration first, then a discrete search pass that
can optimize the exact Minecraft-executable model.

## Suggested Repository Structure

```text
MCNN/
  README.md
  pyproject.toml
  src/
    mcnn/
      __init__.py
      constants.py
      layers.py
      quantization.py
      models.py
      export.py
  experiments/
    basic_task/
      train.py
      evaluate.py
      evolve.py
      config.yaml
  tests/
    test_quantization.py
    test_exact_inference.py
```

Suggested responsibilities:

- `src/mcnn/constants.py`: default bounds for weights, outputs, and bias.
- `src/mcnn/layers.py`: PyTorch modules for constrained linear layers and
  bounded integer activations.
- `src/mcnn/quantization.py`: STE quantizers for weights, biases, and
  activations.
- `src/mcnn/models.py`: small network definitions built from constrained
  layers.
- `src/mcnn/export.py`: conversion from trained PyTorch modules to the exact
  Minecraft-compatible representation.
- `experiments/basic_task/train.py`: fast training loop for a simple supervised
  task such as boolean functions, small grid classification, or toy control.
- `experiments/basic_task/evaluate.py`: exact quantized evaluation with no STE
  behavior.
- `experiments/basic_task/evolve.py`: evolutionary refinement over the exported
  discrete network.
- `tests/test_quantization.py`: checks that all exported weights and outputs
  stay within their configured bounds and all biases are positive.
- `tests/test_exact_inference.py`: checks that PyTorch quantized inference and
  exported inference agree.

## First Basic ML Task

A good first task is a small boolean or grid-world perception problem where the
target function is known and evaluation is cheap.

Examples:

- Learn `AND`, `OR`, `NAND`, `XOR`, and small composed boolean circuits.
- Classify tiny binary grids, such as detecting whether a block pattern contains
  a vertical line, corner, or enclosed cell.
- Map a small local Minecraft observation window to a simple action such as
  `move_forward`, `turn_left`, or `jump`.

Start with boolean tasks because they are easy to exhaustively test. Move to
small grid tasks once quantization, exact inference, export, and refinement are
working.

For a tiny Minecraft integration fixture with binary inputs and redstone signal
outputs, use:

```bash
.venv/bin/python experiments/basic_task/train_binary_integration.py
```

This trains on all `32` possible 5-bit inputs, saves
`artifacts/mcnn_binary_5x5_integration.pt`, and exports
`artifacts/mcnn_binary_5x5_integration.schem`. The default target is:

```text
y0 = clamp(1 + 2*x0 + 4*x1 + 8*x2, 0, 15)
y1 = clamp(15 - 3*x0 - 3*x1 - 3*x2 - 3*x3 - 2*x4, 0, 15)
y2 = clamp(2 + 3*x0 + 3*x1 + 3*x2 + 3*x3 + 3*x4, 0, 15)
y3 = clamp(5 + 5*x0 - 4*x1 + 3*x2 - 2*x3 + x4, 0, 15)
y4 = clamp(1 + 14*x4, 0, 15)
```

For a deterministic multilayer wiring fixture, use:

```bash
.venv/bin/python experiments/basic_task/train_binary_multilayer_integration.py
```

This exports a `5 -> 4 -> 5` model to
`artifacts/mcnn_binary_5x4x5_integration.schem`. Its hidden layer is:

```text
h0 = clamp(x0, 0, 1)
h1 = clamp(x1, 0, 1)
h2 = clamp(x2, 0, 1)
h3 = clamp(x3 + x4, 0, 1)
```

The output layer is:

```text
y0 = clamp(1 + 14*h0, 0, 15)
y1 = clamp(15 - 14*h1, 0, 15)
y2 = clamp(2 + 6*h0 - 4*h2, 0, 15)
y3 = clamp(3 + 12*h3, 0, 15)
y4 = clamp(5 + 5*h0 - 4*h1 + 3*h2 - 2*h3, 0, 15)
```

## Iteration Loop

1. Train with STE in PyTorch.
2. Quantize during every forward pass.
3. Regularly evaluate exact discrete inference.
4. Export the best quantized model.
5. Run evolutionary refinement on the exported model.
6. Re-evaluate the refined model with the exact Minecraft rules.
7. Only then translate or emit the Minecraft implementation.

The main invariant is that final reported performance must come from exact
discrete inference, not from latent floating-point parameters.

## Current 8x8 Digits Baseline

The basic experiment trains on scikit-learn's built-in 8x8 handwritten digit
dataset. Input pixels are binarized before entering the network:

```text
x = 1 if pixel > threshold else 0
```

Minecraft-executable multilayer models must use binary hidden activations:

```text
h = 1 if pre_activation > 0 else 0
```

Only the final layer emits `0..15` redstone signal strengths. The current best
Minecraft-representable run uses:

```bash
.venv/bin/python experiments/basic_task/train.py \
  --epochs 80 \
  --batch-size 64 \
  --hidden 64 \
  --hidden-activation binary_step \
  --learning-rate 0.02 \
  --threshold 4 \
  --loss mse \
  --validation-fraction 0.2 \
  --ablation-passes 1 \
  --ablation-accuracy-tolerance 0.0 \
  --evolution-generations 10 \
  --evolution-population 24 \
  --evolution-elites 4 \
  --evolution-weight-mutations 24 \
  --evolution-bias-mutations 4 \
  --evolution-zero-probability 0.15 \
  --evolution-remove-unit-probability 0.1 \
  --seed 1 \
  --save-path artifacts/mcnn_digits_8x8_best_minecraft.pt
```

Current regenerated metrics:

- Layer sizes: `64 -> 48 -> 10`.
- Exact validation accuracy: `0.937`.
- Reserved test accuracy: `0.882`.
- Nonzero weights: `3,253`.
- Maximum physical bias: `816`.

## Schematic Export

The first schematic exporter is a dense diagonally staggered debug layout. Each
perceptron is a horizontal row with signal-strength weight sources every second
block, and each next output row is one block higher and shifted one block along
X. Neural layers continue on that same diagonal; at a layer boundary, an output
comparator points directly into each next-layer bias barrel. Layers are stacked
after each other along Z with two fully empty Z slices between layers for now.
Exact inter-layer signal extraction and fanout wiring are still left for later.

```bash
.venv/bin/python experiments/basic_task/export_schematic.py \
  --artifact artifacts/mcnn_digits_8x8_best_minecraft.pt \
  --output artifacts/mcnn_digits_8x8_best_minecraft_layers_with_io_wiring.schem \
  --include-input-wiring \
  --include-output-wiring \
  --include-layer-io-connectors
```

Current export summary for the binary-hidden artifact:

- Layer sizes: `64 -> 48 -> 10`.
- Layer 0: `3,072` input slots, max physical bias `816`.
- Layer 1: `480` input slots, max physical bias `616`.

Zero weights use Mode C and omit the stored-weight branch. Nonzero weights are
encoded as Mode A or Mode B modules with a furnace holding the stored `1..14`
signal strength. Bias sources are emitted as physical-bias barrels, so their
stored signal may exceed `15` and uses glitched stacks of unstackable items. The
post-training repair, ablation, and evolution stages reject models whose
physical bias values cannot be represented by this barrel encoding. The redstone
dust above each exposed weight furnace is capped with concrete to avoid
unwanted dust connections. The control block behind the weight furnace is a torch
for Mode A and a repeater for Mode B.
