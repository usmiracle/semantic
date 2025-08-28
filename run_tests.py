#!/usr/bin/env python3
"""
Test runner for the semantic similarity program.
This script provides an easy way to run tests with different options.
"""

import sys
import os
import subprocess
import argparse

def run_unit_tests(verbose=False):
    """Run the unit tests."""
    print("="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run tests
    import test_semantic
    
    # Run tests with appropriate verbosity
    verbosity = 2 if verbose else 1
    result = test_semantic.unittest.main(
        module=test_semantic,
        verbosity=verbosity,
        exit=False
    )
    
    return result.result.wasSuccessful()

def run_performance_test():
    """Run the performance test with real model."""
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE TEST")
    print("="*60)
    
    try:
        # Import the performance test function
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import test_semantic
        test_semantic.run_performance_test()
        return True
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False

def run_coverage_test():
    """Run tests with coverage reporting."""
    print("="*60)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*60)
    
    try:
        # Check if coverage is installed
        import coverage
    except ImportError:
        print("Coverage not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
    
    # Run coverage
    subprocess.check_call([
        sys.executable, "-m", "coverage", "run", "--source=semantic", 
        "-m", "unittest", "test_semantic"
    ])
    
    # Generate report
    subprocess.check_call([sys.executable, "-m", "coverage", "report"])
    
    # Generate HTML report
    subprocess.check_call([sys.executable, "-m", "coverage", "html"])
    print("\nHTML coverage report generated in htmlcov/ directory")

def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running specific test: {test_name}")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_semantic
    
    # Create test suite with specific test
    loader = test_semantic.unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name, test_semantic)
    
    runner = test_semantic.unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for semantic similarity program')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--performance', action='store_true', help='Run performance test')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--test', type=str, help='Run specific test by name')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no arguments provided, run unit tests by default
    if not any([args.unit, args.performance, args.coverage, args.test, args.all]):
        args.unit = True
    
    success = True
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        if args.unit or args.all:
            success = run_unit_tests(args.verbose)
        
        if args.performance or args.all:
            perf_success = run_performance_test()
            success = success and perf_success
        
        if args.coverage or args.all:
            run_coverage_test()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 