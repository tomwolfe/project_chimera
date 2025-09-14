# src/resilience/circuit_breaker.py
import time
from functools import wraps
import logging
from typing import Callable, Any, Type, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Custom exception for when a circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    A circuit breaker implementation to prevent repeated calls to a failing service.
    States:
    - CLOSED: Allows operations. If failure threshold is met, transitions to OPEN.
    - OPEN: Rejects operations immediately. If recovery timeout passes, transitions to HALF-OPEN.
    - HALF-OPEN: Allows a single operation. If it succeeds, transitions to CLOSED. If it fails, transitions back to OPEN.
    """

    def __init__(
        self,
        failure_threshold=5,  # Changed from 3 to 5 as per suggestion
        recovery_timeout=60,
        expected_exception: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # Can be "CLOSED", "OPEN", "HALF_OPEN"
        self.recovery_attempts = 0
        self.logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Check if circuit breaker allows execution."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.recovery_attempts += 1
                self.logger.info(
                    f"Circuit breaker transitioning from OPEN to HALF-OPEN (attempt {self.recovery_attempts})."
                )
                return True  # Allow one attempt in HALF-OPEN
            return False
        return self.state == "HALF_OPEN"  # Allow one attempt in HALF_OPEN

    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker OPENED after {self.failures} failures."
            )

    def record_success(self):
        """Record a successful call and reset the circuit."""
        self.failures = 0
        self.state = "CLOSED"
        self.logger.info("Circuit breaker transitioned to CLOSED after success.")

    def __call__(self, func: Callable) -> Callable:
        """Make this class usable as a decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_available():
                self.logger.warning(
                    f"Circuit breaker is OPEN. Rejecting call to {func.__name__}."
                )
                raise CircuitBreakerError(
                    f"Service unavailable: Circuit breaker is open for {func.__name__}."
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except self.expected_exception as e:
                self.record_failure()
                self.logger.error(
                    f"Call to {func.__name__} failed. Current state: {self.state}.",
                    exc_info=True,
                )
                raise  # Re-raise the original exception
            except Exception as e:
                # For unexpected exceptions, we also count as failures
                self.record_failure()
                self.logger.error(
                    f"Call to {func.__name__} failed with unexpected error. Current state: {self.state}.",
                    exc_info=True,
                )
                raise

        return wrapper