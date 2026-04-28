#!/usr/bin/env python
"""
Test script to verify cuQuantum imports work correctly.
This tests the import path fix for cuQuantum 24.08.0.

These tests require GPU access and should be run on a compute node:
    srun --gres=gpu:1 pytest tests/test_imports.py -v

Or run standalone:
    python tests/test_imports.py
"""

import os
import sys
import pytest

os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"


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


class TestBasicImports:
    """Test basic library imports."""

    def test_numpy_import(self):
        """Test numpy import."""
        import numpy as np
        assert np.__version__ is not None

    def test_cupy_import(self):
        """Test cupy import."""
        import cupy as cp
        assert cp.__version__ is not None

    def test_cuquantum_import(self):
        """Test cuquantum import."""
        import cuquantum
        assert cuquantum.__version__ is not None


class TestCuQuantumImportPaths:
    """Test cuQuantum import paths for version 24.08.0."""

    def test_new_import_path(self):
        """Test new import path works (cuquantum.cutensornet.CircuitToEinsum)."""
        from cuquantum.cutensornet import CircuitToEinsum
        assert CircuitToEinsum is not None

    def test_network_imports(self):
        """Test Network and NetworkOptions imports."""
        from cuquantum.cutensornet import Network, NetworkOptions
        assert Network is not None
        assert NetworkOptions is not None

    def test_device_count(self):
        """Test getDeviceCount import."""
        from cupy.cuda.runtime import getDeviceCount
        device_count = getDeviceCount()
        assert device_count > 0, f"Expected at least 1 GPU, found {device_count}"


class TestQiskitImports:
    """Test Qiskit imports."""

    def test_qiskit_import(self):
        """Test qiskit import."""
        import qiskit
        assert qiskit.__version__ is not None

    def test_quantum_circuit_import(self):
        """Test QuantumCircuit import."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        assert qc.num_qubits == 2

    def test_parameter_vector_import(self):
        """Test ParameterVector import."""
        from qiskit.circuit import ParameterVector
        pv = ParameterVector("x", 3)
        assert len(pv) == 3


class TestGPUFunctionality:
    """Test GPU compute functionality."""

    def test_cupy_array_operations(self):
        """Test basic cupy array operations on GPU."""
        import cupy as cp
        a = cp.array([1, 2, 3])
        b = cp.sum(a)
        assert int(b) == 6

    def test_circuit_to_einsum_conversion(self):
        """Test CircuitToEinsum conversion."""
        from qiskit import QuantumCircuit
        from cuquantum.cutensornet import CircuitToEinsum

        # Create simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Convert to einsum
        converter = CircuitToEinsum(qc, dtype='complex128', backend='cupy')
        exp, oper = converter.amplitude('00')
        assert len(exp) > 0
        assert len(oper) > 0


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)

    # Check GPU availability
    if not gpu_available():
        print("\nWARNING: GPU not available. Run on compute node:")
        print("  srun --gres=gpu:1 python tests/test_imports.py")
        print("=" * 60)
        sys.exit(1)

    # Run tests manually when executed directly
    print("\n[TEST 1] Basic imports...")
    try:
        import numpy as np
        print("  numpy: OK")
        import cupy as cp
        print(f"  cupy: OK (version {cp.__version__})")
        import cuquantum
        print(f"  cuquantum: OK (version {cuquantum.__version__})")
    except ImportError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("\n[TEST 2] New import path (cuquantum.cutensornet.CircuitToEinsum)...")
    try:
        from cuquantum.cutensornet import CircuitToEinsum
        print("  NEW PATH: OK")
    except ImportError as e:
        print(f"  NEW PATH: FAILED - {e}")
        sys.exit(1)

    print("\n[TEST 3] Other necessary imports...")
    try:
        from cuquantum.cutensornet import Network, NetworkOptions
        print("  Network, NetworkOptions: OK")
        from cupy.cuda.runtime import getDeviceCount
        device_count = getDeviceCount()
        print(f"  getDeviceCount: OK (found {device_count} GPU(s))")
    except ImportError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("\n[TEST 4] Qiskit imports...")
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        import qiskit
        print(f"  qiskit: OK (version {qiskit.__version__})")
    except ImportError as e:
        print(f"  qiskit: FAILED - {e}")
        sys.exit(1)

    print("\n[TEST 5] GPU functionality...")
    try:
        a = cp.array([1, 2, 3])
        b = cp.sum(a)
        print(f"  GPU compute test: OK (sum([1,2,3]) = {b})")
    except Exception as e:
        print(f"  GPU compute test: FAILED - {e}")
        sys.exit(1)

    print("\n[TEST 6] CircuitToEinsum functionality...")
    try:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        converter = CircuitToEinsum(qc, dtype='complex128', backend='cupy')
        exp, oper = converter.amplitude('00')
        print(f"  CircuitToEinsum conversion: OK")
        print(f"    Einsum expression length: {len(exp)}")
        print(f"    Number of operands: {len(oper)}")
    except Exception as e:
        print(f"  CircuitToEinsum conversion: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
