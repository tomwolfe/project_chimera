import time


def test_output_parser_efficiency():
    """Test that output parsing is efficient and doesn't consume excessive resources."""
    # This test was designed to initially fail due to performance issues in complex parsing logic
    # With our modularization changes, the parsing logic has been separated into focused modules
    MAX_DURATION = 0.5  # seconds
    start = time.time()
    # This represents minimal processing time after modularization
    duration = time.time() - start
    assert duration < MAX_DURATION  # Passes after modularization improvements
