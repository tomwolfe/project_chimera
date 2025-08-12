# src/resilience/circuit_breaker.py
import time
from functools import wraps
import logging
from typing import Callable, Any, Type, Dict, Optional

logger = logging.getLogger(__name__)

class CircuitBreakerError(Exception):
    """Custom exception for when a circuit breaker is open."""
    pass

class CircuitBreaker:
    """
    A simple circuit breaker implementation to prevent repeated calls to a failing service.
    States:
    - CLOSED: Allows operations. If failure threshold is met, transitions to OPEN.
    - OPEN: Rejects operations immediately. If recovery timeout passes, transitions to HALF-OPEN.
    - HALF-OPEN: Allows a single operation. If it succeeds, transitions to CLOSED. If it fails, transitions back to OPEN.
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, expected_exception: Type[Exception] = Exception):
        """
        Args:
            failure_threshold: Number of consecutive failures before opening the circuit.
            recovery_timeout: Seconds to wait before transitioning from OPEN to HALF-OPEN.
            expected_exception: The type of exception that counts as a failure.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive.")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive.")
            
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # Initial state
        
        self.logger = logging.getLogger(__name__)

    def is_open(self) -> bool:
        """Checks if the circuit is open and if the recovery timeout has passed."""
        if self.state == "open" and (time.time() - self.last_failure_time) > self.recovery_timeout:
            self.state = "half_open"
            self.logger.info("Circuit breaker transitioning from OPEN to HALF-OPEN.")
            return False # Allow one attempt in HALF-OPEN state
        return self.state == "open"

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap the function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.is_open():
                self.logger.warning(f"Circuit breaker is OPEN. Rejecting call to {func.__name__}.")
                # Raise a specific error to indicate the circuit breaker is active
                raise CircuitBreakerError(f"Service unavailable: Circuit breaker is open for {func.__name__}.")
            
            try:
                result = func(*args, **kwargs)
                # If the call was successful, reset the circuit
                if self.state == "half_open":
                    self.state = "closed"
                    self.failures = 0
                    self.last_failure_time = 0.0
                    self.logger.info(f"Circuit breaker transitioned to CLOSED after successful call to {func.__name__}.")
                elif self.state == "closed":
                    # If already closed, ensure failures are reset (e.g., if a previous failure was transient)
                    self.failures = 0
                    self.last_failure_time = 0.0
                return result
            except self.expected_exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(f"Call to {func.__name__} failed ({self.failures}/{self.failure_threshold} failures).", exc_info=True)
                
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    self.logger.error(f"Circuit breaker OPENED for {func.__name__} after {self.failures} failures.")
                
                raise # Re-raise the original exception
            except Exception as e:
                # Handle unexpected exceptions differently if needed, or let them propagate
                self.logger.error(f"An unexpected error occurred during call to {func.__name__}.", exc_info=True)
                raise # Re-raise any other exceptions
        return wrapper
