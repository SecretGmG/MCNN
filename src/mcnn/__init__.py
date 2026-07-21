"""Minecraft-constrained neural network components."""

from .constants import (
    DEFAULT_OUTPUT_MAX,
    DEFAULT_OUTPUT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_WEIGHT_MIN,
    MAX_BIAS_SIGNAL_STRENGTH,
)
from .discrete import (
    DiscreteLayer,
    DiscreteMCNN,
    discrete_from_state_dict,
    remove_hidden_unit,
)
from .encoding import (
    ComparatorEncoding,
    ComparatorSubtractions,
    comparator_subtractions,
    encode_comparator_weight,
    inactive_subtraction,
)
from .export import (
    build_schematic,
    save_schematic,
)
from .quantization import (
    bounded_integer_ste,
    exact_bounded_integer,
    positive_bias,
    quantize_weight,
)
from .refinement import (
    RefinementConfig,
    discrete_score,
    evaluate_discrete,
    evolutionary_refinement,
    local_search_refinement,
    mutate_discrete,
    mutate_local_search_candidate,
    repair_physical_bias_cap,
    satisfies_physical_bias_cap,
    structural_ablation,
)
from .trainable import TrainableMCNN

__all__ = [
    "DEFAULT_OUTPUT_MAX",
    "DEFAULT_OUTPUT_MIN",
    "DEFAULT_WEIGHT_MAX",
    "DEFAULT_WEIGHT_MIN",
    "MAX_BIAS_SIGNAL_STRENGTH",
    "DiscreteLayer",
    "DiscreteMCNN",
    "ComparatorEncoding",
    "ComparatorSubtractions",
    "TrainableMCNN",
    "RefinementConfig",
    "bounded_integer_ste",
    "comparator_subtractions",
    "discrete_from_state_dict",
    "build_schematic",
    "encode_comparator_weight",
    "exact_bounded_integer",
    "inactive_subtraction",
    "positive_bias",
    "quantize_weight",
    "remove_hidden_unit",
    "repair_physical_bias_cap",
    "satisfies_physical_bias_cap",
    "discrete_score",
    "evaluate_discrete",
    "evolutionary_refinement",
    "local_search_refinement",
    "mutate_discrete",
    "mutate_local_search_candidate",
    "structural_ablation",
    "save_schematic",
]
