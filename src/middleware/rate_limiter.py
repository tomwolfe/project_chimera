# src/middleware/rate_limiter.py
import time
from collections import defaultdict
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional, List, Tuple # Import List and Tuple

logger = logging.getLogger(__name__)

class RateLimitExceededError(Exception):
    """Custom exception for rate limit exceeded."""
    pass

class RateLimiter:
    """
    A simple in-memory rate limiter based on user/session identifiers.
    Limits the number of calls within a specified time period.
    """
    
    def __init__(self, key_func: Callable, calls: int = 10, period: float = 60.0):
        """
        Args:
            key_func: A function that returns a unique identifier for the caller (e.g., session ID, IP).
            calls: Maximum number of calls allowed within the period.
            period: The time period in seconds.
        """
        if calls <= 0:
            raise ValueError("calls must be positive.")
        if period <= 0:
            raise ValueError("period must be positive.")
            
        self.calls = calls
        self.period = period
        self.key_func = key_func # Store the provided key_func
        self.timestamps: Dict[str, List[float]] = defaultdict(list)
        self.warning_threshold = 0.7 # NEW: Threshold for warning (70% of limit)
        
        self.logger = logging.getLogger(__name__)

    def check_and_record_call(self, key: str, dry_run: bool = False) -> Tuple[bool, int, float, float]:
        """
        Checks if a call is allowed, records it if not a dry run, and returns usage stats.
        Returns: (is_allowed, current_count, time_to_wait, usage_percent)
        """
        now = time.time()
        
        # Clean up old timestamps for this key
        self.timestamps[key] = [ts for ts in self.timestamps[key] 
                               if now - ts < self.period]
        
        current_count = len(self.timestamps[key])
        usage_percent = (current_count / self.calls) * 100
        
        # Check if the rate limit is exceeded
        if current_count >= self.calls:
            oldest_ts = self.timestamps[key][0]
            remaining_time = self.period - (now - oldest_ts)
            return False, current_count, remaining_time, usage_percent
        
        if not dry_run:
            # Record the current call timestamp
            self.timestamps[key].append(now)
            self.logger.debug(f"Call recorded for key '{key}'. Count: {len(self.timestamps[key])}/{self.calls} in {self.period}s period.")
            current_count += 1 # Increment for the newly recorded call
            usage_percent = (current_count / self.calls) * 100 # Recalculate usage percent
        
        return True, current_count, 0.0, usage_percent

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self.key_func() # Use the key_func provided during instantiation
            is_allowed, count, wait_time, usage_percent = self.check_and_record_call(key, dry_run=False) # Not a dry run for actual calls
            
            if not is_allowed:
                error_msg = f"Rate limit exceeded for key '{key}'. Please try again in {wait_time:.1f} seconds."
                self.logger.warning(error_msg)
                raise RateLimitExceededError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper