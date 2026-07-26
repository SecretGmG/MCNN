# Shift Robustness Findings

The Minecraft digit inputs are not expected to be perfectly aligned, so the
important target is robustness to small non-cropping shifts, not just accuracy
on centered curated digits.

## Evaluation Sets

- Original curated dataset: 3,504 samples
- Shifted dataset: 13,507 samples generated from all valid shifts up to +/-2
  pixels without cropping active pixels

## Main Tradeoff

| Model | Original accuracy | Shifted accuracy |
| --- | ---: | ---: |
| No augmentation | 0.977 | 0.471 |
| Shift-only training | 0.881 | 0.852 |
| Best shift sweep | 0.898 | 0.868 |

The no-augmentation model is excellent for centered curated inputs, but fails
on shifted inputs. Shift-only augmentation is the correct training target for
Minecraft usage.

## Best Current Configuration

| Setting | Value |
| --- | --- |
| Architecture | `[64, 64, 10]` |
| Augmentation | valid shifts only |
| Bit flips | disabled |
| Learning rate | `0.008` |
| Signal penalty weight | `0.02` |
| Signal penalty threshold | `850` |
| Best epoch | `45` |
| Validation accuracy | `0.903` |
| Original accuracy | `0.898` |
| Shifted accuracy | `0.868` |
| Max signal | `897 / 897` |
| Signal violations | `0` |

This model is physically valid but uses the full available signal range.

## Physicality Penalty

Relaxing the physicality penalty improved accuracy:

| Run | Original | Shifted | Max signal |
| --- | ---: | ---: | ---: |
| weight `0.02`, threshold `850` | 0.898 | 0.868 | 897 |
| weight `0.10`, threshold `750` | 0.882 | 0.856 | 886 |
| weight `0.05`, threshold `800` | 0.857 | 0.828 | 897 |

The relaxed setting is currently best, but the stronger setting keeps more
signal margin.

## Depth And Identity Warmup

Deeper `[64, 64, 64, 10]` models did not beat the shallow model. Identity
warmup did help information flow through the deeper network:

| Run | Original | Shifted | Max signal |
| --- | ---: | ---: | ---: |
| deep, no warmup | 0.843 | 0.818 | 897 |
| deep, 5-epoch identity warmup | 0.856 | 0.839 | 870 |
| deep, 10-epoch identity warmup | 0.851 | 0.790 | 868 |

Identity warmup is promising, but depth is not yet worth the extra circuit cost.

## Evolution

Evolution on the best sweep candidate found some higher internal scores, but
the selected candidate was rejected by the validation floor. It did not improve
the saved model in this run.

## Recommendation

Use the best shift-sweep model as the current Minecraft-facing model. Future
work should prioritize:

- evaluating selection directly on shifted validation accuracy,
- more targeted evolution scored on shifted robustness,
- convolution-like feature extraction for translation robustness.
