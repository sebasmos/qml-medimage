#!/usr/bin/env python
"""
Test that the main scripts can import correctly after the fix.

These tests require GPU access and should be run on a compute node:
    srun --gres=gpu:1 pytest tests/test_script_imports.py -v

Or run standalone:
    python tests/test_script_imports.py
"""

import os
import sys
import pytest

os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gpu_available():
    """Check if GPU is available and usable."""
    try:
        import cupy as cp
        # Check if cupy has the basic array attribute (not just installed but functional)
        if not hasattr(cp, 'array'):
            return False
        from cupy.cuda.runtime import getDeviceCount
        if getDeviceCount() == 0:
            return False
        # Try to actually allocate on GPU to verify it's usable
        a = cp.array([1, 2, 3])
        _ = cp.sum(a)
        return True
    except Exception:
        return False


# Mark all tests in this module as requiring GPU
pytestmark = pytest.mark.skipif(
    not gpu_available(),
    reason="GPU not available - run on compute node with GPU"
)


class TestScriptImports:
    """Test that main script imports work correctly."""

    def test_core_imports(self):
        """Test core library imports used by scripts."""
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import joblib
        assert np is not None
        assert pd is not None

    def test_sklearn_imports(self):
        """Test scikit-learn imports."""
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, precision_score, f1_score, recall_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        assert SVC is not None

    def test_qiskit_imports(self):
        """Test Qiskit imports."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        assert QuantumCircuit is not None
        assert ParameterVector is not None

    def test_cuquantum_imports(self):
        """Test cuQuantum imports with new paths."""
        from cuquantum.cutensornet import Network, NetworkOptions, CircuitToEinsum
        from cupy.cuda.runtime import getDeviceCount
        assert CircuitToEinsum is not None
        assert Network is not None

    def test_qve_imports(self):
        """Test qve module imports."""
        from qve import (
            set_seed,
            data_prepare_cv,
            make_bsp,
            build_qsvm_qc,
            data_partition,
            data_to_operand,
            operand_to_amp,
            get_kernel_matrix
        )
        assert set_seed is not None
        assert make_bsp is not None


class TestCircuitCreation:
    """Test quantum circuit creation and conversion."""

    def test_make_bsp(self):
        """Test make_bsp circuit creation."""
        import numpy as np
        from qve import make_bsp

        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        assert bsp_qc is not None

    def test_build_qsvm_qc(self):
        """Test QSVM circuit building."""
        import numpy as np
        from qve import make_bsp, build_qsvm_qc

        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        test_data = np.array([0.5, 0.3])
        circuit = build_qsvm_qc(bsp_qc, n_qubits, test_data, test_data)
        assert circuit is not None

    def test_circuit_to_einsum_conversion(self):
        """Test full circuit to einsum pipeline."""
        import numpy as np
        from cuquantum.cutensornet import CircuitToEinsum
        from qve import make_bsp, build_qsvm_qc

        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        test_data = np.array([0.5, 0.3])
        circuit = build_qsvm_qc(bsp_qc, n_qubits, test_data, test_data)

        converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
        a = str(0).zfill(n_qubits)
        exp, oper = converter.amplitude(a)
        assert len(exp) > 0
        assert len(oper) > 0


class TestNetworkSetup:
    """Test cuQuantum network setup."""

    def test_network_creation(self):
        """Test Network creation and contraction path."""
        import numpy as np
        from cuquantum.cutensornet import Network, NetworkOptions, CircuitToEinsum
        from qve import make_bsp, build_qsvm_qc

        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        test_data = np.array([0.5, 0.3])
        circuit = build_qsvm_qc(bsp_qc, n_qubits, test_data, test_data)

        converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
        a = str(0).zfill(n_qubits)
        exp, oper = converter.amplitude(a)

        device_id = 0
        options = NetworkOptions(blocking="auto", device_id=device_id)
        network = Network(exp, *oper, options=options)
        path, info = network.contract_path()
        assert path is not None


if __name__ == "__main__":
    print("Testing script imports after fix...")
    print("=" * 60)

    if not gpu_available():
        print("\nWARNING: GPU not available. Run on compute node:")
        print("  srun --gres=gpu:1 python tests/test_script_imports.py")
        print("=" * 60)
        sys.exit(1)

    # Test 1: Main insurance script imports
    print("\n[TEST 1] qsvm_cuda_embeddings_insurance.py imports...")
    try:
        import numpy as np
        import pandas as pd
        import cupy as cp
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import joblib
        from mpi4py import MPI

        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, precision_score, f1_score, recall_score,
            roc_auc_score, confusion_matrix, classification_report
        )

        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        from cuquantum.cutensornet import Network, NetworkOptions, CircuitToEinsum
        from cupy.cuda.runtime import getDeviceCount

        from itertools import combinations, product

        from qve import (
            set_seed,
            data_prepare_cv,
            make_bsp,
            build_qsvm_qc,
            data_partition,
            data_to_operand,
            operand_to_amp,
            get_kernel_matrix
        )

        print("  All imports: OK")
    except ImportError as e:
        print(f"  Import FAILED: {e}")
        sys.exit(1)

    # Test 2: Create simple quantum circuit like the script does
    print("\n[TEST 2] Circuit creation and conversion...")
    try:
        n_qubits = 2
        bsp_qc = make_bsp(n_qubits)
        print(f"  make_bsp({n_qubits}): OK")

        test_data = np.array([0.5, 0.3])
        circuit = build_qsvm_qc(bsp_qc, n_qubits, test_data, test_data)
        print(f"  build_qsvm_qc: OK")

        converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy')
        a = str(0).zfill(n_qubits)
        exp, oper = converter.amplitude(a)
        print(f"  CircuitToEinsum conversion: OK")
        print(f"    Expression: {exp[:50]}...")
        print(f"    Operands: {len(oper)} tensors")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 3: Network setup
    print("\n[TEST 3] Network setup...")
    try:
        device_id = 0
        options = NetworkOptions(blocking="auto", device_id=device_id)
        network = Network(exp, *oper, options=options)
        path, info = network.contract_path()
        print(f"  Network creation: OK")
        print(f"  Contract path found: OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("The scripts should now work correctly with cuQuantum 24.08.0")
    print("=" * 60)
