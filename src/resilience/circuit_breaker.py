# src/resilience/circuit_breaker.py
import time
from functools import wraps
import logging
from typing import Callable, Any, Type, Dict, Optional

logger = logging.getLogger(__name__)

class CircuitBreakerError(Exception):
    """Custom exception for when a circuit breaker is open."""
    pass

class AdaptiveCircuitBreaker: # RENAMED CLASS
    """
    A simple circuit breaker implementation to prevent repeated calls to a failing service.
    States:
    - CLOSED: Allows operations. If failure threshold is met, transitions to OPEN.
    - OPEN: Rejects operations immediately. If recovery timeout passes, transitions to HALF-OPEN.
    - HALF-OPEN: Allows a single operation. If it succeeds, transitions to CLOSED. If it fails, transitions back to OPEN.
    """
    
    def __init__(self, failure_threshold=5, recovery_timeout=30, sliding_window_size=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.sliding_window_size = sliding_window_size
        self.failures = [] # Stores timestamps of failures
        self.state = 'CLOSED'
        self.last_failure_time = 0
        self.recovery_attempts = 0
        self.logger = logging.getLogger(__name__)

    def record_failure(self):
        current_time = time.time()
        # Keep only failures within the sliding window
        self.failures = [t for t in self.failures if current_time - t < self.sliding_window_size]
        self.failures.append(current_time)
        self.last_failure_time = current_time
        
        # Calculate failure rate in the sliding window (optional, but good for adaptive logic)
        # failure_rate = len(self.failures) / self.sliding_window_size # Not directly used in this snippet, but good to keep

        if len(self.failures) >= self.failure_threshold:
            self.state = 'OPEN'
            self.recovery_attempts = 0 # Reset attempts when opening
            self.logger.warning(f"Circuit breaker OPENED after {len(self.failures)} failures.")
    
    def record_success(self):
        if self.state == 'HALF_OPEN':
            # Reset circuit if success in half-open state
            self.state = 'CLOSED'
            self.failures = []
            self.recovery_attempts = 0
            self.logger.info("Circuit breaker transitioned to CLOSED after success in HALF-OPEN state.")
        elif self.state == 'CLOSED':
            # If already closed, just clear old failures to keep list clean
            current_time = time.time()
            self.failures = [t for t in self.failures if current_time - t < self.sliding_window_size]

    def is_available(self):
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure > self._calculate_dynamic_timeout():
                self.state = 'HALF_OPEN'
                self.recovery_attempts += 1 # Increment attempts for next OPEN state
                self.logger.info(f"Circuit breaker transitioning from OPEN to HALF-OPEN (attempt {self.recovery_attempts}).")
                return True # Allow one attempt in HALF-OPEN
            return False
        return self.state == 'HALF_OPEN' # Allow one attempt in HALF-OPEN

    def _calculate_dynamic_timeout(self):
        # Exponential backoff for recovery attempts
        base_timeout = self.recovery_timeout
        # Cap exponential backoff to prevent excessively long timeouts
        return base_timeout * (2 ** min(self.recovery_attempts, 5)) # Cap at 2^5 * base_timeout

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap the function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_available():
                self.logger.warning(f"Circuit breaker is OPEN. Rejecting call to {func.__name__}.")
                raise CircuitBreakerError(f"Service unavailable: Circuit breaker is open for {func.__name__}.")
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except self.expected_exception as e: # Assuming expected_exception is still defined in __init__
                self.record_failure()
                self.logger.error(f"Call to {func.__name__} failed. Current state: {self.state}.", exc_info=True)
                raise # Re-raise the original exception
            except Exception as e:
                # Catch all other unexpected errors as failures too
                self.record_failure()
                self.logger.error(f"An unexpected error occurred during call to {func.__name__}. Current state: {self.state}.", exc_info=True)
                raise # Re-raise any other exceptions
        return wrapper