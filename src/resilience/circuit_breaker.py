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
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, 
                 expected_exception: Type[Exception] = Exception,
                 min_failure_threshold: int = 2, max_failure_threshold: int = 10,
                 history_window_size: int = 20): # Track last N calls for adaptive logic
        """
        Args:
            failure_threshold: Number of consecutive failures before opening the circuit.
            recovery_timeout: Seconds to wait before transitioning from OPEN to HALF-OPEN.
            expected_exception: The type of exception that counts as a failure.
            min_failure_threshold: Minimum allowed dynamic failure threshold.
            max_failure_threshold: Maximum allowed dynamic failure threshold.
            history_window_size: Number of recent calls to consider for adaptive threshold adjustment.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive.")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive.")
            
        self.base_failure_threshold = failure_threshold # Store initial default
        self.current_failure_threshold = failure_threshold # This will be dynamic
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.min_failure_threshold = min_failure_threshold
        self.max_failure_threshold = max_failure_threshold
        self.history_window_size = history_window_size

        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # Initial state
        self.call_history = [] # Stores (timestamp, success_boolean) for adaptive logic
        
        self.logger = logging.getLogger(__name__)

    def _record_call(self, success: bool):
        """Records a call's outcome and updates adaptive metrics."""
        current_time = time.time()
        self.call_history.append((current_time, success))
        
        # Keep only calls within the history window
        # Prune older entries to maintain window size and relevance
        self.call_history = [c for c in self.call_history if current_time - c[0] < self.recovery_timeout * 2] # Keep relevant history
        if len(self.call_history) > self.history_window_size:
            self.call_history = self.call_history[-self.history_window_size:]

        self._adjust_threshold()

    def _adjust_threshold(self):
        """Dynamically adjusts the failure threshold based on recent performance."""
        if len(self.call_history) < self.history_window_size / 2: # Need enough data to make a decision
            return

        recent_failures = sum(1 for _, s in self.call_history if not s)
        failure_rate = recent_failures / len(self.call_history)

        if failure_rate < 0.1: # Very low failure rate, increase tolerance
            self.current_failure_threshold = min(self.max_failure_threshold, self.current_failure_threshold + 1)
            self.logger.debug(f"Failure rate low ({failure_rate:.2f}), increasing threshold to {self.current_failure_threshold}")
        elif failure_rate < 0.3:  # NEW: Handle moderate failure rates (0.1 to 0.3)
            # Gradually reduce threshold, but not as aggressively as high failure rates.
            # This provides a "grace period" for temporary spikes without immediately opening the circuit.
            if self.current_failure_threshold > self.min_failure_threshold:
                self.current_failure_threshold = max(self.min_failure_threshold,
                                                  self.current_failure_threshold - 0.5) # Use 0.5 step for gradual adjustment
            self.logger.debug(f"Moderate failure rate ({failure_rate:.2f}), gentle threshold adjustment to {self.current_failure_threshold}")
        elif failure_rate > 0.3: # High failure rate, decrease tolerance
            self.current_failure_threshold = max(self.min_failure_threshold, self.current_failure_threshold - 1)
            self.logger.debug(f"Failure rate high ({failure_rate:.2f}), decreasing threshold to {self.current_failure_threshold}")
        # Otherwise, keep current threshold

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
                self._record_call(True) # Record success
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
                self._record_call(False) # Record failure
                self.logger.error(f"Call to {func.__name__} failed ({self.failures}/{self.current_failure_threshold} failures).", exc_info=True)
                
                if self.failures >= self.current_failure_threshold:
                    self.state = "open"
                    self.logger.error(f"Circuit breaker OPENED for {func.__name__} after {self.failures} failures.")
                
                raise # Re-raise the original exception
            except Exception as e:
                # Handle unexpected exceptions differently if needed, or let them propagate
                self.logger.error(f"An unexpected error occurred during call to {func.__name__}.", exc_info=True)
                self._record_call(False) # Record failure for unexpected errors too
                raise # Re-raise any other exceptions
        return wrapper