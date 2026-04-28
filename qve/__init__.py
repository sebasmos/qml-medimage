# Re-export useful public API
from .utils import set_seed
from .process import process_folds, data_prepare, data_prepare_cv
from .metrics import (
    get_metrics_multiclass_case,
    get_metrics_multiclass_case_cv,
    get_metrics_multiclass_case_test,
)
from .core import (
    make_bsp,
    make_bsp_reps,
    make_bsp_3dof,
    make_zz_featuremap,
    compute_zz_kernel_entries,
    build_qsvm_qc,
    sin_cos,
    get_from_d1,
    get_from_d2,
    renew_operand,
    renew_operand_reps,
    renew_operand_3dof,
    data_partition,
    data_to_operand,
    data_to_operand_reps,
    data_to_operand_3dof,
    operand_to_amp,
    get_kernel_matrix,
    get_hybrid_kernel_matrix,
    normalize_kernel_trace,
    normalize_kernel_frobenius,
    normalize_kernel_cosine,
    compute_projected_features,
    projected_kernel_matrix,
)
