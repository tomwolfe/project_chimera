# src/database/db_operations.py
import sqlite3
from typing import Any, Dict, Optional


# Placeholder for a database connection function
def get_db_connection():
    """Establishes and returns a SQLite database connection."""
    # In a real application, this would connect to a more robust database
    # and handle connection pooling.
    conn = sqlite3.connect("chimera.db")
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def get_user_data(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user data from the database using parameterized queries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Use parameterized queries to prevent SQL injection
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    return dict(user_data) if user_data else None


def update_user_profile(user_id: int, profile_data: Dict[str, Any]):
    """Updates user profile information using parameterized queries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Construct parameterized query safely
    placeholders = ", ".join([f"{key} = ?" for key in profile_data])
    query = f"UPDATE users SET {placeholders} WHERE id = ?"
    params = list(profile_data.values()) + [user_id]
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    return True


def create_users_table():
    """Creates a simple users table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            profile_json TEXT
        )
    """)
    conn.commit()
    conn.close()


# Call to create table on module import (for simple setup)
create_users_table()
