#!/usr/bin/env python3
"""
Standalone test runner for qve module - no pytest required.

Usage:
    python tests/run_tests_standalone.py
"""

import os
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASSED = 0
FAILED = 0
SKIPPED = 0
GPU_FAILED = 0  # Failed due to missing GPU deps


def test(name, requires_gpu=False):
    """Decorator for test functions."""
    def decorator(func):
        global PASSED, FAILED, SKIPPED, GPU_FAILED
        try:
            func()
            print(f"  ✓ PASS: {name}")
            PASSED += 1
        except ModuleNotFoundError as e:
            if requires_gpu and ('cuquantum' in str(e) or 'cupy' in str(e)):
                print(f"  ○ SKIP: {name} (requires GPU: {e})")
                SKIPPED += 1
            else:
                print(f"  ✗ FAIL: {name}")
                print(f"    Error: {e}")
                FAILED += 1
        except Exception as e:
            print(f"  ✗ FAIL: {name}")
            print(f"    Error: {e}")
            traceback.print_exc()
            FAILED += 1
        return func
    return decorator


def import_qve_core():
    """Import qve.core directly without going through __init__.py (avoids torch dependency)."""
    import importlib.util
    core_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "qve", "core.py"
    )
    spec = importlib.util.spec_from_file_location("qve_core", core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_qve_core_tests():
    """Run tests for qve/core.py module."""
    print("\n" + "=" * 60)
    print("TESTING: qve/core.py")
    print("=" * 60)

    @test("Core module has no syntax errors")
    def test_syntax():
        import py_compile
        core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qve", "core.py"
        )
        py_compile.compile(core_path, doraise=True)

    @test("Core module imports successfully (direct import)", requires_gpu=True)
    def test_imports():
        core = import_qve_core()
        assert core is not None

    @test("All functions importable from qve.core", requires_gpu=True)
    def test_all_functions():
        core = import_qve_core()
        assert hasattr(core, 'make_bsp')
        assert hasattr(core, 'build_qsvm_qc')
        assert hasattr(core, 'get_kernel_matrix')
        assert hasattr(core, 'get_hybrid_kernel_matrix')
        assert hasattr(core, 'data_partition')

    @test("make_bsp creates valid circuit", requires_gpu=True)
    def test_make_bsp():
        core = import_qve_core()
        qc = core.make_bsp(2)
        assert qc.num_qubits == 2

    @test("make_bsp creates parameterized circuit", requires_gpu=True)
    def test_make_bsp_params():
        core = import_qve_core()
        qc = core.make_bsp(3)
        assert qc.num_parameters == 3

    @test("build_qsvm_qc creates kernel circuit", requires_gpu=True)
    def test_build_qsvm():
        core = import_qve_core()
        bsp_qc = core.make_bsp(2)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.2, 0.8])
        circuit = core.build_qsvm_qc(bsp_qc, 2, x1, x2)
        assert circuit.num_qubits == 2

    @test("data_partition single process", requires_gpu=True)
    def test_partition_single():
        core = import_qve_core()
        indices = list(range(10))
        result = core.data_partition(indices, size=1, rank=0)
        assert result == indices

    @test("data_partition multi-process", requires_gpu=True)
    def test_partition_multi():
        core = import_qve_core()
        indices = list(range(10))
        rank0 = core.data_partition(indices, size=2, rank=0)
        rank1 = core.data_partition(indices, size=2, rank=1)
        assert len(rank0) + len(rank1) == len(indices)
        assert set(rank0).isdisjoint(set(rank1))

    @test("get_kernel_matrix correct shape", requires_gpu=True)
    def test_kernel_shape():
        core = import_qve_core()
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        amp_data = [[1.0, 0.8, 0.6, 1.0, 0.7, 1.0]]
        kernel = core.get_kernel_matrix(data, data, amp_data, indices, mode='train')
        assert kernel.shape == (3, 3)

    @test("get_kernel_matrix symmetric in train mode", requires_gpu=True)
    def test_kernel_symmetric():
        core = import_qve_core()
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        amp_data = [[1.0, 0.8, 0.6, 1.0, 0.7, 1.0]]
        kernel = core.get_kernel_matrix(data, data, amp_data, indices, mode='train')
        np.testing.assert_array_almost_equal(kernel, kernel.T)

    @test("get_kernel_matrix diagonal is 1 in train mode", requires_gpu=True)
    def test_kernel_diagonal():
        core = import_qve_core()
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        amp_data = [[0.5, 0.8, 0.6, 0.5, 0.7, 0.5]]  # off-diag values
        kernel = core.get_kernel_matrix(data, data, amp_data, indices, mode='train')
        np.testing.assert_array_almost_equal(np.diag(kernel), np.ones(3))


def run_hybrid_kernel_tests():
    """Run tests for hybrid kernel functionality."""
    print("\n" + "=" * 60)
    print("TESTING: Hybrid Kernel Functionality")
    print("=" * 60)

    @test("get_hybrid_kernel_matrix alpha=0 gives pure classical", requires_gpu=True)
    def test_alpha_zero():
        core = import_qve_core()
        K_c = np.array([[1, 0.5], [0.5, 1]])
        K_q = np.array([[1, 0.3], [0.3, 1]])
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=0.0)
        np.testing.assert_array_almost_equal(K_h, K_c)

    @test("get_hybrid_kernel_matrix alpha=1 gives pure quantum", requires_gpu=True)
    def test_alpha_one():
        core = import_qve_core()
        K_c = np.array([[1, 0.5], [0.5, 1]])
        K_q = np.array([[1, 0.3], [0.3, 1]])
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=1.0)
        np.testing.assert_array_almost_equal(K_h, K_q)

    @test("get_hybrid_kernel_matrix alpha=0.5 gives correct mix", requires_gpu=True)
    def test_alpha_half():
        core = import_qve_core()
        K_c = np.array([[1, 0.5], [0.5, 1]])
        K_q = np.array([[1, 0.3], [0.3, 1]])
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=0.5)
        expected = 0.5 * K_q + 0.5 * K_c
        np.testing.assert_array_almost_equal(K_h, expected)

    @test("get_hybrid_kernel_matrix preserves shape", requires_gpu=True)
    def test_shape():
        core = import_qve_core()
        K_c = np.random.randn(10, 10)
        K_q = np.random.randn(10, 10)
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=0.7)
        assert K_h.shape == K_c.shape

    @test("get_hybrid_kernel_matrix works with non-square matrices", requires_gpu=True)
    def test_non_square():
        core = import_qve_core()
        K_c = np.random.randn(5, 10)
        K_q = np.random.randn(5, 10)
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=0.3)
        assert K_h.shape == (5, 10)

    @test("Hybrid of symmetric matrices is symmetric", requires_gpu=True)
    def test_symmetry():
        core = import_qve_core()
        K_c = np.random.randn(4, 4)
        K_c = (K_c + K_c.T) / 2
        K_q = np.random.randn(4, 4)
        K_q = (K_q + K_q.T) / 2
        K_h = core.get_hybrid_kernel_matrix(K_c, K_q, alpha=0.6)
        np.testing.assert_array_almost_equal(K_h, K_h.T)
    
    @test("normalize_kernel_cosine works with square matrices", requires_gpu=True)
    def test_cosine_square():
        core = import_qve_core()
        K = np.array([[1.0, 0.8, 0.6],
                      [0.8, 1.0, 0.7],
                      [0.6, 0.7, 1.0]])
        K_norm = core.normalize_kernel_cosine(K)
        assert K_norm.shape == K.shape
        # Check diagonal is all 1s after cosine normalization
        np.testing.assert_array_almost_equal(np.diag(K_norm), np.ones(3))
    
    @test("normalize_kernel_cosine handles rectangular matrices (bug fix)", requires_gpu=True)
    def test_cosine_rectangular():
        """This test verifies the fix for the broadcasting error."""
        core = import_qve_core()
        # Rectangular matrix (like validation/test kernel)
        K_rect = np.array([[0.9, 0.8, 0.6, 0.5, 0.4],
                           [0.7, 0.8, 0.7, 0.6, 0.5]])
        # This should not crash!
        K_norm = core.normalize_kernel_cosine(K_rect)
        assert K_norm.shape == K_rect.shape
        # For rectangular matrices, should return unchanged
        np.testing.assert_array_equal(K_norm, K_rect)
    
    @test("normalize_kernel_trace works with square matrices", requires_gpu=True)
    def test_trace_square():
        core = import_qve_core()
        K = np.array([[2.0, 0.5], [0.5, 2.0]])
        K_norm = core.normalize_kernel_trace(K)
        assert K_norm.shape == K.shape
        assert abs(np.trace(K_norm) - 1.0) < 1e-10
    
    @test("normalize_kernel_trace returns unchanged for rectangular matrices", requires_gpu=True)
    def test_trace_rectangular():
        core = import_qve_core()
        K_rect = np.array([[0.9, 0.8, 0.6],
                           [0.7, 0.8, 0.7]])
        K_norm = core.normalize_kernel_trace(K_rect)
        np.testing.assert_array_equal(K_norm, K_rect)
    
    @test("normalize_kernel_centered works with square matrices", requires_gpu=True)
    def test_centered_square():
        core = import_qve_core()
        K = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.4],
                      [0.3, 0.4, 1.0]])
        K_norm = core.normalize_kernel_centered(K)
        assert K_norm.shape == K.shape
    
    @test("normalize_kernel_centered returns unchanged for rectangular matrices", requires_gpu=True)
    def test_centered_rectangular():
        core = import_qve_core()
        K_rect = np.array([[0.9, 0.8, 0.6],
                           [0.7, 0.8, 0.7]])
        K_norm = core.normalize_kernel_centered(K_rect)
        np.testing.assert_array_equal(K_norm, K_rect)
    
    @test("normalize_kernel_frobenius works with any matrix", requires_gpu=True)
    def test_frobenius_any():
        core = import_qve_core()
        # Square - use rand for positive values
        K_square = np.random.rand(3, 3)
        K_norm = core.normalize_kernel_frobenius(K_square)
        assert K_norm.shape == K_square.shape
        # Rectangular - use rand for positive values
        K_rect = np.random.rand(2, 5)
        K_norm = core.normalize_kernel_frobenius(K_rect)
        assert K_norm.shape == K_rect.shape
    
    @test("hybrid kernel with cosine normalization - realistic dimensions (bug fix)", requires_gpu=True)
    def test_hybrid_cosine_realistic():
        """This test verifies the fix with realistic dimensions from the error log."""
        core = import_qve_core()
        # From error: (200,1600) and (200,200)
        # Training: 1600 samples, Test/Val: 200 samples
        n_train = 100  # Reduced for testing
        n_test = 20
        
        # Training kernels (square)
        K_train_q = np.random.rand(n_train, n_train)
        K_train_c = np.random.rand(n_train, n_train)
        
        # Test kernels (rectangular)
        K_test_q = np.random.rand(n_test, n_train)
        K_test_c = np.random.rand(n_test, n_train)
        
        alpha = 0.5
        
        # This should not crash!
        K_train_hybrid = core.get_hybrid_kernel_matrix(K_train_c, K_train_q, alpha, "cosine")
        assert K_train_hybrid.shape == (n_train, n_train)
        
        K_test_hybrid = core.get_hybrid_kernel_matrix(K_test_c, K_test_q, alpha, "cosine")
        assert K_test_hybrid.shape == (n_test, n_train)



def run_script_isolation_tests():
    """Run tests to verify script isolation."""
    print("\n" + "=" * 60)
    print("TESTING: Script Isolation")
    print("=" * 60)

    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"
    )

    @test("qsvm_hybrid_insurance.py exists")
    def test_hybrid_exists():
        path = os.path.join(scripts_dir, "qsvm_hybrid_insurance.py")
        assert os.path.exists(path)

    @test("qsvm_hybrid_insurance.py has valid syntax")
    def test_hybrid_syntax():
        import py_compile
        path = os.path.join(scripts_dir, "qsvm_hybrid_insurance.py")
        py_compile.compile(path, doraise=True)

    @test("qsvm_cuda_embeddings_insurance.py has valid syntax")
    def test_main_syntax():
        import py_compile
        path = os.path.join(scripts_dir, "qsvm_cuda_embeddings_insurance.py")
        py_compile.compile(path, doraise=True)

    @test("qsvm_hybrid_insurance.py defines own get_hybrid_kernel_matrix")
    def test_hybrid_local_func():
        import ast
        path = os.path.join(scripts_dir, "qsvm_hybrid_insurance.py")
        with open(path, 'r') as f:
            tree = ast.parse(f.read())
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert 'get_hybrid_kernel_matrix' in func_names

    @test("qsvm_cuda_embeddings_insurance.py has apply_hybrid_kernel with passthrough")
    def test_main_passthrough():
        import ast
        path = os.path.join(scripts_dir, "qsvm_cuda_embeddings_insurance.py")
        with open(path, 'r') as f:
            content = f.read()
        assert 'if not use_hybrid:' in content
        assert 'return kernel_train, kernel_val' in content

    @test("Main script --use_hybrid defaults to False (store_true)")
    def test_default_false():
        path = os.path.join(scripts_dir, "qsvm_cuda_embeddings_insurance.py")
        with open(path, 'r') as f:
            content = f.read()
        assert 'action="store_true"' in content or "action='store_true'" in content


def run_backwards_compatibility_tests():
    """Run backwards compatibility tests."""
    print("\n" + "=" * 60)
    print("TESTING: Backwards Compatibility")
    print("=" * 60)

    @test("All original core functions available (direct import)", requires_gpu=True)
    def test_original_exports():
        core = import_qve_core()
        required_funcs = [
            'make_bsp', 'build_qsvm_qc', 'data_partition',
            'data_to_operand', 'operand_to_amp', 'get_kernel_matrix',
        ]
        for func in required_funcs:
            assert hasattr(core, func), f"Missing function: {func}"

    @test("New hybrid export available (direct import)", requires_gpu=True)
    def test_new_export():
        core = import_qve_core()
        assert hasattr(core, 'get_hybrid_kernel_matrix')

    @test("get_kernel_matrix works without hybrid features", requires_gpu=True)
    def test_kernel_independent():
        core = import_qve_core()
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        amp_data = [[1.0, 0.8, 0.6, 1.0, 0.7, 1.0]]
        kernel = core.get_kernel_matrix(data, data, amp_data, indices, mode='train')
        assert kernel.shape == (3, 3)


def main():
    global PASSED, FAILED, SKIPPED

    print("=" * 60)
    print("QML-MedImage Test Suite (Standalone)")
    print("=" * 60)
    print("Testing qve module and hybrid functionality")
    print("Note: Some tests require GPU (cuquantum) - run on compute node")

    run_qve_core_tests()
    run_hybrid_kernel_tests()
    run_script_isolation_tests()
    run_backwards_compatibility_tests()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed:  {PASSED}")
    print(f"  Skipped: {SKIPPED} (require GPU)")
    print(f"  Failed:  {FAILED}")
    print("=" * 60)

    if FAILED > 0:
        print("\n⚠️  SOME TESTS FAILED!")
        sys.exit(1)
    elif SKIPPED > 0 and PASSED > 0:
        print("\n✅ NON-GPU TESTS PASSED!")
        print("   Run on GPU node for full test coverage:")
        print("   srun --gres=gpu:1 python tests/run_tests_standalone.py")
        sys.exit(0)
    else:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
