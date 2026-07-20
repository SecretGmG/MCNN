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
    discrete_from_model,
    discrete_from_state_dict,
    inactive_subtraction,
    remove_hidden_unit,
)
from .export import (
    ExportPlan,
    SchematicExportConfig,
    build_module_atlas,
    build_schematic,
    export_plan_from_discrete,
    is_representable_bias_signal,
    save_module_atlas,
    save_schematic,
    signal_strength_barrel,
    signal_strength_furnace,
)
from .layers import BinaryStepActivation, BoundedIntegerActivation, MCNNLayer
from .models import MCNNClassifier
from .quantization import (
    ComparatorEncoding,
    ComparatorSubtractions,
    bounded_integer_ste,
    comparator_subtractions,
    encode_comparator_weight,
    exact_bounded_integer,
    positive_bias,
    quantize_weight,
)

__all__ = [
    "BoundedIntegerActivation",
    "BinaryStepActivation",
    "DEFAULT_OUTPUT_MAX",
    "DEFAULT_OUTPUT_MIN",
    "DEFAULT_WEIGHT_MAX",
    "DEFAULT_WEIGHT_MIN",
    "MAX_BIAS_SIGNAL_STRENGTH",
    "DiscreteLayer",
    "DiscreteMCNN",
    "ExportPlan",
    "SchematicExportConfig",
    "ComparatorEncoding",
    "ComparatorSubtractions",
    "MCNNClassifier",
    "MCNNLayer",
    "bounded_integer_ste",
    "comparator_subtractions",
    "discrete_from_model",
    "discrete_from_state_dict",
    "export_plan_from_discrete",
    "is_representable_bias_signal",
    "build_module_atlas",
    "build_schematic",
    "encode_comparator_weight",
    "exact_bounded_integer",
    "inactive_subtraction",
    "positive_bias",
    "quantize_weight",
    "remove_hidden_unit",
    "save_module_atlas",
    "save_schematic",
    "signal_strength_barrel",
    "signal_strength_furnace",
]
