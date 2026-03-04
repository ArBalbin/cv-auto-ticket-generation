# app/auth_service.py
"""
Authentication Service
======================
Handles all user authentication logic, keeping app.py clean.
"""

import mysql.connector
from functools import wraps
from flask import session, jsonify, redirect, url_for, flash


class AuthService:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    # =========================================================================
    # USER VERIFICATION
    # =========================================================================

    def verify_user(self, username: str, password: str) -> tuple[bool, str]:
        """
        Verify username and password against the database.

        Returns:
            (True, "") on success
            (False, reason_string) on failure
        """
        if not username or not password:
            return False, "Username and password are required."

        if not self.db_pool:
            return False, "Database unavailable. Please try again later."

        conn = cursor = None
        try:
            conn   = self.db_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT password FROM users WHERE username = %s LIMIT 1",
                (username,)
            )
            row = cursor.fetchone()

            if row is None:
                return False, "Invalid username or password."

            if row['password'] != password:
                return False, "Invalid username or password."

            return True, ""

        except mysql.connector.errors.PoolExhausted:
            return False, "Server is busy. Please try again shortly."

        except mysql.connector.OperationalError as e:
            print(f"❌ Auth DB connection error: {e}")
            return False, "Database connection failed. Please try again."

        except mysql.connector.Error as e:
            print(f"❌ Auth DB error: {e}")
            return False, "An internal error occurred. Please try again."

        finally:
            if cursor:
                try: cursor.close()
                except Exception: pass
            if conn:
                try: conn.close()
                except Exception: pass

    def login_user(self, username: str, password: str) -> tuple[bool, str]:
        """Verify and set session. Returns (success, error_message)."""
        ok, err = self.verify_user(username, password)
        if ok:
            session['user'] = username
        return ok, err

    def logout_user(self):
        session.pop('user', None)

    @staticmethod
    def current_user() -> str | None:
        return session.get('user')

    @staticmethod
    def is_authenticated() -> bool:
        return 'user' in session

    # =========================================================================
    # DECORATORS
    # =========================================================================

    @staticmethod
    def require_session(f):
        """
        Decorator for HTML routes — redirects to login page if not authenticated.
        Usage:
            @app.route('/dashboard')
            @auth.require_session
            def dashboard(): ...
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user' not in session:
                flash('Please login to access this page.', 'warning')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated

    @staticmethod
    def require_api_auth(f):
        """
        Decorator for JSON API routes — returns 401 JSON if not authenticated.
        Usage:
            @app.route('/api/protected')
            @auth.require_api_auth
            def protected_endpoint(): ...
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user' not in session:
                return jsonify({'error': 'Unauthorized', 'message': 'Login required.'}), 401
            return f(*args, **kwargs)
        return decorated