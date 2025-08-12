# src/middleware/rate_limiter.py
import time
from collections import defaultdict
import streamlit as st
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger(__name__)

class RateLimitExceededError(Exception):
    """Custom exception for rate limit exceeded."""
    pass

class RateLimiter:
    """
    A simple in-memory rate limiter based on user/session identifiers.
    Limits the number of calls within a specified time period.
    """
    
    def __init__(self, calls: int = 5, period: float = 60.0, key_func: Optional[Callable] = None):
        """
        Args:
            calls: Maximum number of calls allowed within the period.
            period: The time period in seconds.
            key_func: A function that returns a unique identifier for the caller (e.g., session ID, IP).
                      Defaults to using Streamlit's session ID.
        """
        if calls <= 0:
            raise ValueError("calls must be positive.")
        if period <= 0:
            raise ValueError("period must be positive.")
            
        self.calls = calls
        self.period = period
        self.key_func = key_func or self._default_key_func
        self.timestamps: Dict[str, List[float]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)

    def _default_key_func(self) -> str:
        """Default key function using Streamlit session ID."""
        try:
            # Accessing Streamlit's session state requires the Streamlit context
            # We use a hypothetical '_session_id' key, assuming it might be set by Streamlit
            # or needs to be explicitly managed if not directly available.
            # A more robust approach might involve a custom session management if Streamlit's internal ID isn't stable.
            session_id = st.session_state.get('_session_id', None) 
            if session_id:
                return session_id
            
            # Fallback if session_id is not directly available or set.
            # This might require a more robust way to get a unique identifier per user/session.
            # For simplicity, we'll use a generic identifier if session state is not ready.
            # In a real app, consider using IP address or a unique user ID if authenticated.
            return "anonymous_user_session" 
        except Exception:
            # If Streamlit context is not available (e.g., running outside Streamlit context)
            # or session state is not initialized yet.
            return "no_streamlit_context"

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self.key_func()
            now = time.time()
            
            # Clean up old timestamps for this key
            self.timestamps[key] = [ts for ts in self.timestamps[key] 
                                   if now - ts < self.period]
            
            # Check if the rate limit is exceeded
            if len(self.timestamps[key]) >= self.calls:
                # Calculate remaining time until the oldest timestamp expires
                oldest_ts = self.timestamps[key][0]
                remaining_time = self.period - (now - oldest_ts)
                error_msg = f"Rate limit exceeded for key '{key}'. Please try again in {remaining_time:.1f} seconds."
                self.logger.warning(error_msg)
                raise RateLimitExceededError(error_msg)
            
            # Record the current call timestamp
            self.timestamps[key].append(now)
            self.logger.debug(f"Call recorded for key '{key}'. Count: {len(self.timestamps[key])}/{self.calls} in {self.period}s period.")
            
            return func(*args, **kwargs)
        return wrapper
