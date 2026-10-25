# Minecraft Neural Network

The Minecraft neural network is a streamed, time-multiplexed network with
ternary weights and integer activations. Layer input and output sizes are
padded to multiples of eight.

## Compute cell

One cell evaluates eight neurons. Two neurons are multiplexed spatially in the
upper and lower circuits, and four neuron states are multiplexed in time around
an eight-comparator loop. A new weight word enters every two redstone ticks, so
the loop carries four independent time slots.

Each hexadecimal weight word is decoded to four bits. The two most significant
bits control the upper circuit and the two least significant bits control the
lower circuit. Each bit pair encodes one ternary weight:

| Bits | Weight |
| --- | ---: |
| `00` | 0 |
| `10` | +1 |
| `01` | -1 |
| `11` | forbidden |

Delay lines make both bits of a weight arrive together and offset the upper
circuit by two redstone ticks relative to the lower circuit.

## Neuron operation

During accumulation, integer input activations are streamed through the
activation input while the weight ROM supplies the corresponding ternary
weights. The first loop therefore represents the signed preactivation

```text
s_j = sum_i w[j, i] * x_i,    w[j, i] in {-1, 0, +1}.
```

The physical loop stores this value using an offset, high-range comparator
signal, but software can treat it as a signed integer.

Activation always takes 15 cycles. On cycle `r`, a programmed value `d[r]` is
subtracted from the residual. The second loop records how many resulting
residuals remain positive:

```text
residual[0] = s_j
residual[r] = residual[r - 1] - d[r]
y_j = count(residual[r] > 0 for r = 1..15)
```

Consequently, every output activation is an integer in `[0, 15]`. The 15
subtraction values define a programmable monotone, quantized activation
function and are stored in ROM alongside the weights. The Python model stores
the equivalent cumulative thresholds `t[r] = sum(d[1:r])`, so its exact
activation is simply `count(s_j > t[r])`.

## Layers and connectivity

Outputs are stored in barrels. A downward barrel shifter, with wraparound,
streams them back as inputs to the next layer. A complete rotation implements
a dense layer. Stopping after `k` shifts implements a cyclic block-banded layer
without changing the compute cell.

If cells are indexed modulo the number of input cells, output cell `j` receives
from the `k + 1` consecutive cells

```text
j, j - 1, ..., j - k    (modulo the input cell count).
```

Thus each output neuron has at most `8(k + 1)` inputs. Because the shifter wraps,
there are no boundary cells; the physical downward direction is simply a
choice of cyclic ordering in software. A full dense layer is obtained by
shifting through every input cell.

A fixed tower of compute cells can execute networks of arbitrary depth by
repeating accumulation, 15-cycle activation, output storage, and barrel
shifting for each layer. The network architecture is therefore specified by
the padded layer sizes, the shift count of each layer, the ternary weight ROM,
and the activation-subtraction ROM.
