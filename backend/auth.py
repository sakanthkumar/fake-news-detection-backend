# auth.py
import os
from datetime import datetime, timedelta
from flask import Blueprint, redirect, make_response, current_app
from authlib.integrations.flask_client import OAuth
import jwt
import mysql.connector
import certifi

# OAuth client instance
oauth = OAuth()
auth_bp = Blueprint("auth", __name__)

# ENV variables
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
JWT_SECRET = os.environ.get("JWT_SECRET", "replace_this_secret")
FLASK_ENV = os.environ.get("FLASK_ENV", "development")

# DB config (match your app.py DB config)
# DB config (match your app.py DB config)
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'root'),
    'database': os.environ.get('DB_NAME', 'fake_news_auth'),
    'port': int(os.environ.get('DB_PORT', 3306))
}

if DB_CONFIG['host'] != 'localhost':
    DB_CONFIG['ssl_ca'] = certifi.where()
    DB_CONFIG['ssl_disabled'] = False
    DB_CONFIG['ssl_verify_cert'] = True
    DB_CONFIG['ssl_verify_identity'] = True


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_oauth(app):
    oauth.init_app(app)

    oauth.register(
        name="google",
        client_id=os.environ.get("GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


@auth_bp.route("/google")
def google_login():
    redirect_uri = f"{BACKEND_URL}/auth/google/callback"
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    # Exchange code for token
    try:
        token = oauth.google.authorize_access_token()
    except Exception as e:
        current_app.logger.exception("Google OAuth failed")
        return redirect(f"{FRONTEND_URL}/login")

    # Fetch userinfo
    try:
        userinfo = oauth.google.userinfo()
    except Exception:
        try:
            userinfo = oauth.google.get("userinfo").json()
        except Exception:
            current_app.logger.exception("Google userinfo failed")
            return redirect(f"{FRONTEND_URL}/login")

    uid = userinfo.get("sub")
    email = userinfo.get("email")
    username = userinfo.get("name") or f"google_{uid[:6]}"

    if not uid or not email:
        return redirect(f"{FRONTEND_URL}/login")

    # ---- Find or Create user in DB ----
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # Step 1 — direct match via google provider
        cur.execute("SELECT * FROM users WHERE oauth_provider=%s AND oauth_id=%s LIMIT 1", ("google", uid))
        user = cur.fetchone()

        if not user:
            # Step 2 — match local account via email
            cur.execute("SELECT * FROM users WHERE email=%s LIMIT 1", (email,))
            user = cur.fetchone()

            if user:
                # Merge Google with existing local user
                cur.execute("""
                    UPDATE users SET oauth_provider=%s, oauth_id=%s, is_verified=1 
                    WHERE id=%s
                """, ("google", uid, user["id"]))
                conn.commit()
            else:
                # Step 3 — create new Google user
                cur.execute("""
                    INSERT INTO users (username, email, password, oauth_provider, oauth_id, is_verified)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (username, email, "", "google", uid, 1))
                conn.commit()
                cur.execute("SELECT * FROM users WHERE id=%s", (cur.lastrowid,))
                user = cur.fetchone()

        cur.close()
        conn.close()

    except Exception as e:
        current_app.logger.exception("DB error in google_callback")
        # fallback
        user = {"username": username, "email": email}

    # ---- Create JWT ----
    payload = {
        "sub": uid,
        "username": user["username"],
        "email": user["email"],
        "user_id": user.get("id"),
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=7)
    }

    jwt_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    if isinstance(jwt_token, bytes):
        jwt_token = jwt_token.decode("utf-8")

    # ---- Set cookie ----
    secure_flag = (FLASK_ENV == "production")
    resp = make_response(redirect(f"{FRONTEND_URL}/auth/success"))
    resp.set_cookie(
        "auth_token",
        jwt_token,
        httponly=True,
        secure=secure_flag,
        samesite="Lax",
        max_age=7 * 24 * 60 * 60
    )
    return resp


@auth_bp.route("/logout", methods=["POST"])
def logout():
    resp = make_response({"success": True})
    resp.set_cookie("auth_token", "", expires=0)
    return resp
