# src/database_operations.py
import logging
import sqlite3  # Assuming SQLite for database operations, adjust if different

logger = logging.getLogger(__name__)

# --- Database Connection ---
# In a real application, this would be more robust, possibly using a connection pool
# or managing connections more carefully. For this example, we'll use a simple global.
# NOTE: This is a placeholder and might need adjustment based on actual DB setup.
DB_PATH = "project_chimera.db"  # Example database path


def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise


# --- Database Operations ---


def create_tables():
    """Creates necessary database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                display_name TEXT,
                bio TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        logger.info("Database tables ensured to exist.")
    except sqlite3.Error as e:
        logger.error(f"Error creating database tables: {e}")
        conn.rollback()
    finally:
        conn.close()


def add_user(username, email, password_hash):
    """Adds a new user to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        user_id = cursor.lastrowid
        conn.commit()
        logger.info(f"User '{username}' added successfully with ID: {user_id}")
        return user_id
    except sqlite3.IntegrityError:
        logger.warning(f"Username '{username}' or email '{email}' already exists.")
        conn.rollback()
        return None
    except sqlite3.Error as e:
        logger.error(f"Error adding user '{username}': {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def get_user_by_username(username: str):
    """Retrieves a user by their username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        return user
    except sqlite3.Error as e:
        logger.error(f"Error retrieving user '{username}': {e}")
        return None
    finally:
        conn.close()


def get_user_data(username: str):
    """Retrieves user data including profile information."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Join users and user_profiles tables
        cursor.execute(
            """
            SELECT u.*, up.display_name, up.bio
            FROM users u
            LEFT JOIN user_profiles up ON u.id = up.user_id
            WHERE u.username = ?
        """,
            (username,),
        )
        user_data = cursor.fetchone()
        return user_data
    except sqlite3.Error as e:
        logger.error(f"Error retrieving user data for '{username}': {e}")
        return None
    finally:
        conn.close()


def update_user_profile(user_id: int, profile_data: Dict[str, Any]):
    """Updates a user's profile information."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if profile exists, if not, create it
        cursor.execute(
            "SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        profile_exists = cursor.fetchone()

        if profile_exists:
            # Update existing profile
            update_fields = []
            values = []
            if "display_name" in profile_data:
                update_fields.append("display_name = ?")
                values.append(profile_data["display_name"])
            if "bio" in profile_data:
                update_fields.append("bio = ?")
                values.append(profile_data["bio"])

            if update_fields:
                # Use parameterized queries to prevent SQL injection
                placeholders = ', '.join([f'{field} = ?' for field in update_fields])
                query = f"UPDATE user_profiles SET {placeholders} WHERE user_id = ?"
                params = values + [user_id]
                cursor.execute(query, params)
                conn.commit()
                logger.info(f"User profile updated for user ID: {user_id}")
            else:
                # If no fields to update, just commit the transaction
                pass
        else:
            # Insert new profile
            display_name = profile_data.get("display_name")
            bio = profile_data.get("bio")
            cursor.execute(
                "INSERT INTO user_profiles (user_id, display_name, bio) VALUES (?, ?, ?)",
                (user_id, display_name, bio),
            )
            conn.commit()
            logger.info(f"User profile created for user ID: {user_id}")

    except sqlite3.Error as e:
        logger.error(f"Error updating user profile for user ID {user_id}: {e}")
        conn.rollback()
    finally:
        conn.close()


def delete_user(user_id: int):
    """Deletes a user and their associated profile."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete profile first due to foreign key constraint (if defined)
        cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        # Then delete the user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        logger.info(f"User with ID {user_id} and their profile deleted.")
    except sqlite3.Error as e:
        logger.error(f"Error deleting user with ID {user_id}: {e}")
        conn.rollback()
    finally:
        conn.close()


# --- Example of a function that might be modified by the LLM ---
# This function is hypothetical and serves as an example for the LLM's suggestions.
# The LLM's output suggested modifying this to use parameterized queries.

# Original (hypothetical) vulnerable function:
# def get_user_data_vulnerable(username):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Vulnerable to SQL injection: directly formatting user input into the query
#     cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")
#     user = cursor.fetchone()
#     conn.close()
#     return user


# Modified function (as suggested by LLM):
def get_user_data_secure(username: str):
    """Retrieves user data using parameterized queries for security."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Securely fetch user data using parameterized query
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        return user
    except sqlite3.Error as e:
        logger.error(
            f"Error retrieving user data for '{username}' with parameterized query: {e}"
        )
        return None
    finally:
        conn.close()


# Example of another function that might be modified
# Original (hypothetical) vulnerable function:
# def delete_user_vulnerable(user_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Vulnerable to SQL injection if user_id is not properly sanitized/validated
#     cursor.execute(f"DELETE FROM users WHERE id = {user_id}")
#     conn.commit()
#     conn.close()


# Modified function (as suggested by LLM):
def delete_user_secure(user_id: int):
    """Deletes a user using parameterized queries for security."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Securely delete user using parameterized query
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        logger.info(f"User with ID {user_id} deleted securely.")
    except sqlite3.Error as e:
        logger.error(f"Error deleting user with ID {user_id} securely: {e}")
        conn.rollback()
    finally:
        conn.close()


# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ensure tables exist
    create_tables()

    # Add a user
    user_id = add_user("testuser", "test@example.com", "hashed_password_123")

    if user_id:
        # Update profile
        update_user_profile(
            user_id, {"display_name": "Test User", "bio": "A test user profile."}
        )

        # Get user data
        user_data = get_user_data("testuser")
        if user_data:
            logger.info(f"Retrieved user data: {dict(user_data)}")

        # Get user data using the secure function
        user_data_secure = get_user_data_secure("testuser")
        if user_data_secure:
            logger.info(f"Retrieved user data securely: {dict(user_data_secure)}")

        # Delete user
        delete_user_secure(user_id)

        # Verify deletion
        user_data_after_delete = get_user_data("testuser")
        if user_data_after_delete is None:
            logger.info("User 'testuser' successfully deleted.")
        else:
            logger.warning("User 'testuser' still exists after deletion attempt.")
    else:
        logger.warning("Failed to add user, skipping profile operations.")