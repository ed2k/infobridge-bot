import os
import sys
import importlib.util
import inspect

def run_test_file(filepath):
    print(f"Running test file: {os.path.basename(filepath)}")
    spec = importlib.util.spec_from_file_location("test_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    test_funcs = [func for name, func in inspect.getmembers(module, inspect.isfunction) if name.startswith("test_")]
    
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {func.__name__}")
            failed += 1
        except Exception as e:
            print(f"  ERROR in {func.__name__}: {e}")
            failed += 1
            
    print(f"  Result: {passed} passed, {failed} failed")
    return passed, failed

def main():
    test_dir = "tests"
    all_passed = 0
    all_failed = 0
    for filename in sorted(os.listdir(test_dir)):
        if filename.startswith("test_") and filename.endswith(".py"):
            filepath = os.path.join(test_dir, filename)
            passed, failed = run_test_file(filepath)
            all_passed += passed
            all_failed += failed
            
    print(f"\nTotal Result: {all_passed} passed, {all_failed} failed")
    if all_failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
