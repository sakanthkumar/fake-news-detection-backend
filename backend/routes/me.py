# routes/me.py
import os
from flask import Blueprint, request, jsonify, current_app
import jwt

me_bp = Blueprint("me", __name__)

# Use same algorithm constant used elsewhere
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_SECRET = os.environ.get("JWT_SECRET", None)

@me_bp.route("/me", methods=["GET"])
def me():
    """
    Returns the authenticated user. Supports:
      - Authorization: Bearer <token>
      - httpOnly cookie auth_token
    Response:
      200 -> { success: True, user: {...} }
      401 -> { success: False, error: "..." }
    """
    secret = current_app.config['SECRET_KEY']
    if not secret:
        current_app.logger.error("SECRET_KEY not configured")
        return jsonify({"success": False, "error": "Server misconfigured"}), 500

    # 1) Try Authorization header first
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    # 2) Fallback to cookie
    if not token:
        token = request.cookies.get("auth_token")

    if not token:
        return jsonify({"success": False, "error": "Not authenticated"}), 401

    # Decode token
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return jsonify({"success": False, "error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"success": False, "error": "Invalid token"}), 401
    except Exception as e:
        current_app.logger.exception("Unexpected JWT decode error: %s", e)
        return jsonify({"success": False, "error": "Invalid token"}), 401

    # If token contains user_id, try to fetch full user row from DB.
    user_id = payload.get("user_id")
    if user_id:
        try:
            # Import DB helper inside function to avoid circular imports
            from app import get_db_connection
            conn = get_db_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT id, username, email, oauth_provider, oauth_id, is_verified FROM users WHERE id=%s LIMIT 1",
                (user_id,)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return jsonify({"success": True, "user": row}), 200
            # if row not found, fall back to payload below
        except Exception as e:
            current_app.logger.exception("Failed to fetch user from DB: %s", e)
            # fall back to payload

    # Fallback: return the token payload (non-sensitive fields only)
    safe_user = {
        "username": payload.get("username"),
        "email": payload.get("email"),
        "user_id": payload.get("user_id")
    }
    return jsonify({"success": True, "user": safe_user}), 200
