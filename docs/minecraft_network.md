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
streams them back as inputs to the next layer. Number the logical values from
top to bottom and then by their four temporal slots. Moving down one two-block
slice changes the logical destination by four. For an input width `n`, the
shifter visits matrix diagonals in this exact order:

```text
4, 8, 12, ..., n - 4,
1, 5,  9, ..., n - 3,
2, 6, 10, ..., n - 2,
3, 7, 11, ..., n - 1,
0
```

For diagonal `d`, input value `i` is connected to output neuron
`(i + d) mod n`. The first reachable weight is therefore four rows below the
matrix diagonal, in the next two-block slice. The unshifted diagonal `d = 0` is
the final destination, not the first.

The uppermost and lowermost neuron pairs use the orange-supported endpoint
routes when the stream wraps. The orange blocks are structural and add no
redstone delay themselves. Each endpoint route contains the same three delayed
comparator stages as the ordinary route. At the wrap it advances the temporal
phase, producing the `... n - 4, 1 ...`, `... n - 3, 2 ...` transitions above
without a bubble in the two-RT stream.

Stopping the shifter early retains a prefix of this cyclic-diagonal order. The
current Python configuration is aligned to eight-neuron modules: a
`shift_depth` of `k` retains

```text
w = 8 * (k + 1)
```

diagonals. A full dense layer is obtained when `w = n`. The wrap means there
are no exceptional boundary neurons, but a truncated layer must use the
physical order above rather than an ordinary centered or block-contiguous band.

A fixed tower of compute cells can execute networks of arbitrary depth by
repeating accumulation, 15-cycle activation, output storage, and barrel
shifting for each layer. The network architecture is therefore specified by
the padded layer sizes, the shift count of each layer, the ternary weight ROM,
and the activation-subtraction ROM.

## Exact timing

One redstone tick (RT) is two game ticks, or `0.1 s`. The timing below was
audited against `nn(2).schem` (SHA-256
`a5b5fabf65f0540846fcabdc9c3928e91b98fb7c64f00d37618d565a6084843c`).
It assumes the Java redstone delays represented by the schematic block states.

The barrel shifter has a three-RT longest data path. This includes the three
comparator stages on the orange-supported endpoint route; those stages balance
the corresponding ordinary route. Its four temporal slots remain two RT apart,
so the slots of one cell reach the destination at:

```text
slot 0    slot 1    slot 2    slot 3
  3 RT      5 RT      7 RT      9 RT
```

The fourth slot is visible at absolute time `t = 9 RT` because the whole stream
has a fixed three-RT pipeline latency. The handoff itself nevertheless occupies
exactly eight RT (`0.8 s`), measured from the start of one four-slot cell to the
start of the next. The two height lanes run in parallel. Two four-RT repeaters
in the control path produce this eight-RT transfer window. The shifter's slot
initiation interval remains two RT (`0.2 s`).

For layer `l`, define:

```text
n[l] = padded input width
m[l] = padded output width
k[l] = configured shift depth
w[l] = 8 * (k[l] + 1) = retained physical diagonals
```

The tower contains `m[l] / 8` parallel eight-neuron modules, so output width
changes area and ROM width, not layer latency. Each retained diagonal contributes
eight ternary weights to one output module. A hex word carries two weights, so
one diagonal consumes four words. At two RT per word:

```text
accumulation[l] = 4 * w[l] words * 2 RT = 8 * w[l] RT
activation[l]   = 8 neurons * 15 steps / 2 values-per-word * 2 RT
                = 120 RT
handoff         = 8 RT between consecutive layers
```

For a `D`-layer network whose inputs are already loaded and whose final outputs
are read in place, first-result latency is:

```text
L_first = 8 * sum(w[l]) + 120 * D + 8 * (D - 1) RT
T_first = 0.1 * L_first seconds
```

Add eight RT if inputs must first pass through the shifter, and another eight RT
if final outputs must be handed off instead of read in place. For a dense
layer, `w[l] = n[l]`. For a truncated layer, latency depends only on the number
of retained physical diagonals.

The long first-result latency must not be confused with stream throughput.
After the pipeline is filled, independent examples are separated by two RT.
For a batch of `P` boards evaluated with the same ROM schedule:

```text
L_last(P) = L_first + 2 * (P - 1) RT
throughput = 1 / 0.2 s = 5 evaluations per second
```

The Python model exposes the same calculation as
`MinecraftNetwork.timing(batch_size=...)`.
