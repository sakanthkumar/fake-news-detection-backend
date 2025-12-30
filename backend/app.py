import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Tuple
import re
import joblib, torch, numpy as np, requests, hashlib, jwt, certifi
from transformers import DistilBertTokenizer, DistilBertModel
# from sentence_transformers import SentenceTransformer, util
import mysql.connector
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from functools import wraps
from langdetect import detect, DetectorFactory
from agents import scraper_agent, predictor_agent, explainer_agent, log_agent_start, log_agent_done
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# LIME availability
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

DetectorFactory.seed = 0

# --- App creation & configuration ---
app = Flask(__name__)

# CORS configuration: Allow specific origins (Production Vercel + Local Development)
# Support comma-separated URLs in FRONTEND_URL env var
raw_origins = os.environ.get("FRONTEND_URL", "http://localhost:3000,http://localhost:5173")
origins_list = [origin.strip() for origin in raw_origins.split(",")]
CORS(app, origins=origins_list, supports_credentials=True)

# Use env JWT secret if present; otherwise fall back to hardcoded (avoid in production)
app.config['SECRET_KEY'] = os.environ.get('JWT_SECRET', os.environ.get('SECRET_KEY', 'xg9Bs9B6_T0VMH_D4CGNamuNBTwEelql2uPNxGhx1YQCjSIncPw_UN61CAHJeb2dlDp8H2hQHpGshTKbhNQt7g'))

# For backwards compatibility in your code that referenced JWT_SECRET variable:
JWT_SECRET = os.environ.get('JWT_SECRET', "xg9Bs9B6_T0VMH_D4CGNamuNBTwEelql2uPNxGhx1YQCjSIncPw_UN61CAHJeb2dlDp8H2hQHpGshTKbhNQt7g")
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600  # 1 hour

# --- Initialize OAuth (auth.py must expose init_oauth and auth_bp) ---
# NOTE: auth.py must NOT import `app` (avoid circular import). It should define init_oauth(app), oauth = OAuth(), auth_bp.
try:
    # import the initializer and blueprint from auth.py (expected next to this file)
    from auth import init_oauth, auth_bp
    # bind oauth to the Flask app and register provider(s)
    init_oauth(app)
except Exception as e:
    # If there's an import/initialization failure, log it and continue so it's visible on startup
    print("⚠️  Failed to initialize OAuth (auth.py). Google OAuth will not work until fixed.")
    print(e)
    init_oauth = None
    auth_bp = None

# --- Register other blueprints / routes (me route for cookie-based verification) ---
try:
    from routes.me import me_bp
except Exception as e:
    print("⚠️  Could not import routes.me - /api/me will not be available until fixed.")
    print(e)
    me_bp = None

# Register blueprints only if imported successfully
if auth_bp is not None:
    app.register_blueprint(auth_bp, url_prefix='/auth')

if me_bp is not None:
    app.register_blueprint(me_bp, url_prefix='/api')

# --- Your existing ML / helper initialization ---
# Keep heavy model loading as you had it. If startup is slow, consider lazy-loading these.
# --- ML Global Variables (Lazy Loaded) ---
tokenizer = None
bert_model = None
clf = None
device = torch.device("cpu")

def get_model():
    global tokenizer, bert_model, clf
    if bert_model is None:
        print("⏳ Loading DistilBERT model (Lazy Load)...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        model_fp32 = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        
        # QUANTIZATION: Reduce memory by ~60%
        bert_model = torch.quantization.quantize_dynamic(
            model_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )
        bert_model.eval()
        bert_model.to(device)
        print("✅ DistilBERT Loaded & Quantized.")

    if clf is None:
        try:
            print("⏳ Loading Classifier...")
            clf = joblib.load("distilbert_classifier.pkl")
            print("✅ Classifier Loaded.")
        except Exception as e:
            print(f"❌ Failed to load classifier: {e}")
            clf = None
    
    return tokenizer, bert_model, clf

BANNED_WORDS = []
def contains_inappropriate_content(headline):
    if not headline:
        return False
    return any(word in headline.lower() for word in BANNED_WORDS)

# ===== Semantic Similarity Model (optional, used elsewhere) =====
# ===== Semantic Similarity Model (optional) =====
# REMOVED for Render Free Tier (Memory Optimization)
similarity_model = None
# try:
#     similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
# except Exception as e:
#     similarity_model = None
#     print(f"⚠️ SentenceTransformer load failed: {e}")

# ===== Database Config =====
# ===== Database Config =====
# Read from env vars for Render/Production, fallback to localhost for dev
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'root'),
    'database': os.environ.get('DB_NAME', 'fake_news_auth'),
    'port': int(os.environ.get('DB_PORT', 3306))
}

# TiDB / Remote SSL Config
if DB_CONFIG['host'] != 'localhost':
    DB_CONFIG['ssl_ca'] = certifi.where()
    DB_CONFIG['ssl_disabled'] = False
    DB_CONFIG['ssl_verify_cert'] = True
    DB_CONFIG['ssl_verify_identity'] = True

# ===== Database Initialization =====
def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        c = conn.cursor()
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE,
                email VARCHAR(255) UNIQUE,
                password VARCHAR(255)
            )
        ''')
        # Predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                headline TEXT,
                prediction VARCHAR(255),
                confidence FLOAT,
                username VARCHAR(255),
                timestamp DATETIME
            )
        ''')
        # Feedback table
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                headline TEXT,
                original_prediction VARCHAR(255),
                user_feedback VARCHAR(255),
                username VARCHAR(255),
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB Initialization Failed: {e}")

init_db()

# ... (Helper functions remain same) ...



# ===== Helper Functions =====
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_prediction(headline, prediction, confidence, username=None):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO predictions (headline, prediction, confidence, username, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (str(headline), str(prediction), float(confidence), username, datetime.now())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB Save Failed: {e}")


def transform_headline_single(headline):
    # single headline -> embedding numpy array
    get_model() # Ensure loaded
    tokens = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        output = bert_model(**tokens)
        embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

def batch_transform_headlines(texts):
    # batch transform to embeddings using tokenizer batch mode
    get_model() # Ensure loaded
    if not texts:
        return np.zeros((0, bert_model.config.hidden_size))
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = bert_model(**toks)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
    return emb

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"




def _best_similarity_to_list(headline: str, source_headlines: List[str]) -> float:
    """
    Return best similarity (0..1) between `headline` and any string in `source_headlines`.
    Try SentenceTransformer cosine similarity when available, else fall back to
    simple substring / token-overlap heuristics.
    """
    if not headline or not source_headlines:
        return 0.0

    # 1) sentence-transformer cosine similarity (best when available)
    if similarity_model is not None:
        try:
            q_emb = similarity_model.encode([headline], convert_to_tensor=True)
            s_embs = similarity_model.encode(source_headlines, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, s_embs)[0]
            max_sim = float(sims.max().cpu().numpy())
            return max_sim
        except Exception as e:
            print(f"⚠️ similarity_model failed: {e}")   


    # 2) simple exact / substring match
    ql = headline.lower()
    for s in source_headlines:
        sl = s.lower()
        if ql in sl or sl in ql:
            return 1.0

    # 3) token overlap (Jaccard)
    q_tokens = set(re.findall(r"\w+", ql))
    best = 0.0
    for s in source_headlines:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        if not q_tokens or not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens) / float(len(q_tokens | s_tokens))
        if overlap > best:
            best = overlap
    return best         


# ===== Scraping Helpers (multi-page capable) =====
def scrape_pages(base_url, page_param="page", max_pages=3, selector="h2"):
    headlines = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for p in range(1, max_pages+1):
        # many sites use different pagination patterns; this is a simple attempt
        url = f"{base_url}?{page_param}={p}"
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            items = [h.text.strip() for h in soup.select(selector) if len(h.text.strip()) > 10]
            headlines.extend(items)
        except Exception as e:
            print(f"❌ Failed page {p} for {base_url}: {e}")
    # dedupe
    return list(dict.fromkeys(headlines))

def get_ndtv_headlines(max_pages=3):
    return scrape_pages("https://www.ndtv.com/latest", page_param="page", max_pages=max_pages, selector="h2")

def get_toi_headlines(max_pages=3):
    return scrape_pages("https://timesofindia.indiatimes.com/news", page_param="page", max_pages=max_pages, selector="a")

def get_thehindu_headlines(max_pages=3):
    return scrape_pages("https://www.thehindu.com/news/", page_param="page", max_pages=max_pages, selector="a")

def get_bbc_headlines(max_pages=3):
    return scrape_pages("https://www.bbc.com/news", page_param="page", max_pages=max_pages, selector="h2")

def get_reuters_headlines(max_pages=3):
    # Reuters sometimes uses different param naming; try a default
    return scrape_pages("https://www.reuters.com/news/archive/worldNews", page_param="view", max_pages=max_pages, selector="h3")

def get_cnn_headlines(max_pages=3):
    return scrape_pages("https://edition.cnn.com/world", page_param="page", max_pages=max_pages, selector="span.container__headline-text")

def get_indiatoday_headlines(max_pages=3):
    return scrape_pages("https://www.indiatoday.in/world", page_param="page", max_pages=max_pages, selector="h2")

def get_aljazeera_headlines(max_pages=3):
    return scrape_pages("https://www.aljazeera.com/news/", page_param="page", max_pages=max_pages, selector="h3")

SOURCES = {
    "NDTV": get_ndtv_headlines,
    "TOI": get_toi_headlines,
    "TheHindu": get_thehindu_headlines,
    "BBC": get_bbc_headlines,
    "Reuters": get_reuters_headlines,
    "CNN": get_cnn_headlines,
    "IndiaToday": get_indiatoday_headlines,
    "AlJazeera": get_aljazeera_headlines
}

def fetch_all_headlines_parallel(max_pages=3):
    """Fetch headlines from all SOURCES in parallel."""
    all_headlines = []
    # Reduced workers to prevent OOM (Paging File error)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(func, max_pages=max_pages): name for name, func in SOURCES.items()}
        for future in as_completed(future_to_source):
            name = future_to_source[future]
            try:
                headlines = future.result()
                all_headlines.extend(headlines)
            except Exception as e:
                print(f"❌ {name} failed: {e}")
    return list(dict.fromkeys(all_headlines)) # Dedupe

def check_credibility(headline: str, max_pages: int = 2, threshold: float = 0.70) -> Tuple[List[str], int]:
    """
    Check trusted SOURCES for matching headlines using PARALLEL fetching.
    If that fails, fallback to DuckDuckGo Search to find older/archived news.
    Returns (matched_source_names, credibility_percent).
    Credibility percent = (matches / total_sources) * 100 (integer).
    """
    matched = []
    if not headline:
        return matched, 0

    # 1. Parallel fetch from "Latest" pages (Fast, Real-time)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_source = {executor.submit(func, max_pages=max_pages): name for name, func in SOURCES.items()}
        
        for future in as_completed(future_to_source):
            name = future_to_source[future]
            try:
                src_headlines = future.result()
            except Exception as e:
                print(f"❌ {name} failed during credibility check: {e}")
                src_headlines = []
            
            try:
                best_sim = _best_similarity_to_list(headline, src_headlines)
            except Exception as e:
                print(f"⚠️ similarity check failed for {name}: {e}")
                best_sim = 0.0
            
            if best_sim >= threshold:
                matched.append(name)

    # 2. Fallback: Web Search (DuckDuckGo) if no direct matches found
    # This handles news that is real but not on the "Latest" page (e.g., yesterday's news)
    if not matched:
        try:
            from duckduckgo_search import DDGS
            print(f"🔍 No direct matches. Searching web for: {headline}")
            with DDGS() as ddgs:
                # Search for the headline
                results = list(ddgs.text(headline, max_results=5))
                
                # Check if any result comes from a trusted domain
                trusted_domains = [
                    "ndtv.com", "timesofindia.indiatimes.com", "thehindu.com", 
                    "bbc.com", "reuters.com", "cnn.com", "indiatoday.in", "aljazeera.com"
                ]
                
                for r in results:
                    link = r.get('href', '').lower()
                    title = r.get('title', '').lower()
                    body = r.get('body', '').lower()
                    
                    # Check if link is from a trusted source
                    source_match = next((d for d in trusted_domains if d in link), None)
                    
                    if source_match:
                        # Double check content similarity to ensure it's the same topic
                        # (Simple keyword overlap check)
                        combined_text = title + " " + body
                        if _best_similarity_to_list(headline, [combined_text]) > 0.5:
                            matched.append(f"Web Search ({source_match})")
                            
                # Dedupe
                matched = list(set(matched))
                
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")

    total_sources = max(1, len(SOURCES))
    # If we found it via search, give it a high credibility score (e.g., 80%+)
    if matched and "Web Search" in matched[0]:
        credibility = 85
    else:
        credibility = int(round((len(matched) / total_sources) * 100))
        
    return matched, credibility

def human_readable_explanation(lime_list):
    """
    lime_list: result of exp.as_list(), like [("word", weight), ...] or
    the JSON-style from your lime_explain() function output.
    Return a short sentence summarizing the top positive & negative words.
    """
    if not lime_list:
        return "No explanation available."

    # If lime_list is JSON-friendly dicts (your lime_explain returns that), convert
    if isinstance(lime_list, list) and isinstance(lime_list[0], dict):
        items = lime_list
        pos = [d["word"] for d in items if d["weight"] > 0][:5]
        neg = [d["word"] for d in items if d["weight"] < 0][:5]
    else:
        # assume list of tuples
        items = lime_list
        pos = [w for (w, weight) in items if weight > 0][:5]
        neg = [w for (w, weight) in items if weight < 0][:5]

    parts = []
    if pos:
        parts.append(f"Positive influence from: {', '.join(pos)}")
    if neg:
        parts.append(f"Negative influence from: {', '.join(neg)}")
    return " ; ".join(parts)


# ===== Prediction Core =====
def predict_raw(headline):
    """Basic classifier call (no further verification). Returns (prediction_label, confidence_float)."""
    get_model() # Ensure loaded
    if clf is None:
        return "error", 0.0
    emb = transform_headline_single(headline)
    probas = clf.predict_proba(emb)[0]
    label_index = int(np.argmax(probas))
    prediction = str(clf.classes_[label_index])
    confidence = float(probas[label_index])
    return prediction, confidence

# ===== LIME wrapper =====
def predict_proba_texts(texts):
    """
    Given a list of raw text strings, return predict_proba array from the fitted classifier.
    This is used by LIME which passes perturbed text lists.
    """
    get_model() # Ensure loaded
    if clf is None:
        # return uniform probabilities if clf missing
        n = len(texts)
        if n == 0:
            return np.zeros((0, 2))
        # try to infer classes length; fallback to 2
        try:
            n_classes = len(clf.classes_)
        except Exception:
            n_classes = 2
        return np.ones((n, n_classes)) / float(n_classes)

    # Batch-transform
    emb = batch_transform_headlines(texts)
    try:
        probs = clf.predict_proba(emb.astype(np.float64))
    except Exception as e:
        print(f"❌ clf.predict_proba failed in predict_proba_texts: {e}")
        # fallback: return zeros
        probs = np.zeros((len(texts), len(clf.classes_)))
    return probs

def lime_explain(headline, num_features=6, num_samples=500):
    """
    Return a list of (word, weight) pairs explaining the prediction for `headline`.
    Uses LIME's TextExplainer which perturbs the text and calls predict_proba_texts.
    """
    if not LIME_AVAILABLE:
        raise RuntimeError("LIME is not installed. Please pip install lime.")

    explainer = LimeTextExplainer(class_names=[str(c) for c in (clf.classes_ if clf is not None else ["0","1"])])
    # LIME wants a function f(texts) that returns probability arrays
    f = predict_proba_texts

    # Because LIME's internal tokenization differs, provide appropriate parameters
    try:
        exp = explainer.explain_instance(headline, f, num_features=num_features, num_samples=num_samples)
    except Exception as e:
        print(f"⚠️ explain_instance failed: {e}")
        raise e

    # pick the predicted class
    pred_label, _ = predict_raw(headline)
    
    # find label index in class_names
    try:
        label_index = list(clf.classes_).index(pred_label)
    except Exception:
        # If pred_label is string but classes are ints, try converting
        try:
            label_index = list(clf.classes_).index(int(pred_label))
        except:
            label_index = 0

    print("🔍 Running LIME explanation on:", headline)
    
    try:
        # as_list(label) returns list of (feature, weight)
        lst = exp.as_list(label=label_index)
    except Exception as e:
        print(f"⚠️ exp.as_list failed with label_index={label_index}. Available labels: {exp.available_labels()}")
        raise e

    # convert to JSON-friendly format
    explanation = []
    for feature, weight in lst:
        # feature sometimes like ' word' or 'word'; trim
        explanation.append({
            "word": feature.strip(),
            "weight": float(weight),
            "impact": "supports" if weight > 0 else "contradicts"
        })
    return explanation

# ===== JWT Decorator =====
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Authorization token
        if "Authorization" in request.headers:
            parts = request.headers["Authorization"].split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]

        # Cookie fallback
        if not token:
            token = request.cookies.get("auth_token")

        if not token:
            return jsonify({"error": "Token is missing"}), 401

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            request.username = payload.get("username")
            request.user_id = payload.get("user_id")
        except Exception:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated


# ===== Auth APIs =====
@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = hashlib.sha256(data.get("password", "").encode()).hexdigest()
    if not username or not email or not password:
        return jsonify({"success": False, "error": "username, email and password required"}), 400
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                  (username, email, password))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "User registered successfully!"}), 201
    except mysql.connector.IntegrityError:
        return jsonify({"success": False, "error": "Username or Email already exists"}), 400
    except Exception as e:
        print(f"❌ Register failed: {e}")
        return jsonify({"success": False, "error": "Registration failed"}), 500
    
@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password_raw = data.get("password", "")
    password = hashlib.sha256(password_raw.encode()).hexdigest()
    if not username or not password_raw:
        return jsonify({"success": False, "error": "username and password required"}), 400
    try:
        conn = get_db_connection()
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = c.fetchone()
        conn.close()
        if user and user["password"] == password:
            payload = {
                "username": username,
                "exp": datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
            return jsonify({"success": True, "token": token}), 200
        else:
            return jsonify({"success": False, "error": "Invalid username or password"}), 401
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return jsonify({"success": False, "error": "Login failed"}), 500

# Add this function before the api_predict route
def fetch_prediction_by_headline(headline: str):
    """Fetch existing prediction from database if available"""
    try:
        conn = get_db_connection()
        c = conn.cursor(dictionary=True)
        c.execute("""
            SELECT prediction, confidence, username, timestamp 
            FROM predictions 
            WHERE headline = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (headline,))
        result = c.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"❌ fetch_prediction failed: {e}")
        return None
        


@app.route("/api/predict", methods=["POST"])
@token_required 
def api_predict():
    data = request.json or {}
    headline = (data.get("headline") or "").strip()
    if not headline:
        return jsonify({"error": "headline missing"}), 400
    if contains_inappropriate_content(headline):
        return jsonify({"error": "inappropriate content detected"}), 400

    # Check cache first
    cached = fetch_prediction_by_headline(headline)
    if cached:
        return jsonify({
            "headline": headline,
            "prediction": cached["prediction"],
            "confidence": round(min(cached["confidence"] * 100, 99.99), 2),
            "language": detect_language(headline),
            "cached": True
        })

    try:
        log_agent_start("Predictor Agent", f"Analyzing headline: {headline}")
        
        lang = detect_language(headline)
        prediction, confidence = predict_raw(headline)
        
        if prediction == "error":
            return jsonify({"error": "model not loaded"}), 500

        # Get credibility scores
        matched_sources, credibility = check_credibility(headline, max_pages=2, threshold=0.70)
        
        label = "REAL" if str(prediction) == "1" else "FAKE"

        # --- HYBRID LOGIC: Context-Aware Verification ---
        # Only override Model's REAL verdict if:
        # 1. Credibility is low (not found in trusted sources)
        # 2. AND the headline contains "High Stakes" keywords (death, disaster, etc.)
        # This prevents flagging niche/local news as Fake.
        
        HIGH_STAKES_KEYWORDS = ["dead", "died", "killed", "passed away", "murder", "suicide", "blast", "explosion", "attack", "crash", "fatal"]
        
        is_high_stakes = any(k in headline.lower() for k in HIGH_STAKES_KEYWORDS)
        
        if label == "REAL" and credibility < 20 and is_high_stakes:
            print(f"⚠️ Hybrid Override: Model said REAL but credibility is {credibility}% for High Stakes headline. Flagging as FAKE.")
            label = "FAKE"
            confidence = 0.99 

        # Save new prediction (save the final label, not the raw 0/1)
        save_prediction(headline, label, confidence, request.username)
        
        log_agent_done("Predictor Agent")

        return jsonify({
            "headline": headline,
            "prediction": label,
            "confidence": round(min(confidence * 100, 99.99), 2),
            "language": lang,
            "credibility": credibility,
            "matched_sources": matched_sources,
            "cached": False
        })

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500
    


def map_label(prediction):
    return "REAL" if str(prediction) == "1" else "FAKE"





# ===== API: Explain prediction with LIME =====
@app.route("/api/explain", methods=["POST"])
@token_required
def api_explain():
    if not LIME_AVAILABLE:
        return jsonify({"error": "LIME library is not installed. Run: pip install lime"}), 500

    data = request.json or {}
    headline = data.get("headline", "").strip()
    num_features = int(data.get("num_features", 6))
    num_samples = int(data.get("num_samples", 500))

    if not headline:
        return jsonify({"error": "headline missing"}), 400
    if contains_inappropriate_content(headline):
        return jsonify({"error": "inappropriate content detected"}), 400
    if clf is None:
        return jsonify({"error": "classifier not loaded"}), 500

    # --- Log CrewAI explainer agent ---
    log_agent_start("Explainer Agent", f"Explaining prediction for: {headline}")
    try:
        explainer_agent.run(input=f"Generate explanation for: {headline}")
    except Exception as agent_err:
        print(f"⚠️ CrewAI Explainer Agent failed: {agent_err}")
    log_agent_done("Explainer Agent")

    # --- Language detection ---
    lang = detect_language(headline)

    # --- Model prediction and explanation ---
    try:
        print(f"🔍 Generating prediction and LIME explanation for: {headline}")
        prediction, confidence = predict_raw(headline)

        # ✅ Fix: ensure numeric dtype compatibility for BERT embeddings
        explanation = lime_explain(
            headline, 
            num_features=num_features, 
            num_samples=num_samples
        )

        prediction, confidence = predict_raw(headline)
        label = map_label(prediction)
        raw_lime = lime_explain(headline, num_features=num_features, num_samples=num_samples)
        summary_text = human_readable_explanation(raw_lime)
        save_prediction(headline, label, confidence, request.username)
        return jsonify({
            "headline": headline,
            "prediction": label,
            "confidence": round(min(confidence * 100, 99.99), 2),
            "language": lang,
            "explanation": raw_lime,
            "explanation_summary": summary_text
        })


    except Exception as e:
        print(f"❌ LIME explanation failed: {e}")
        return jsonify({"error": f"LIME explanation failed: {str(e)}"}), 500


# ===== Other APIs (scrape, web predict, fetch-and-save) - unchanged except using helper funcs =====
@app.route("/api/report", methods=["POST"])
@token_required
def api_report():
    data = request.json or {}
    headline = data.get("headline")
    original_prediction = data.get("prediction")
    feedback = data.get("feedback") # e.g., "Incorrect", "Fake", "Real"
    
    if not headline or not feedback:
        return jsonify({"error": "Missing data"}), 400

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO feedback (headline, original_prediction, user_feedback, username, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (headline, original_prediction, feedback, request.username, datetime.now())
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Feedback received. Thank you!"})
    except Exception as e:
        print(f"❌ Report failed: {e}")
        return jsonify({"error": "Failed to save feedback"}), 500

@app.route("/api/scrape", methods=["POST"])
@token_required
def api_scrape():
    data = request.json or {}
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "url missing"}), 400

    log_agent_start("Scraper Agent", f"Scraping latest news from {url}")
    scraper_agent.run(input=f"Scrape latest news headline from {url}")
    log_agent_done("Scraper Agent")

    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        headline = soup.title.string if soup.title else None
    except Exception as e:
        print(f"❌ Scrape single URL failed: {e}")
        headline = None

    if not headline:
        return jsonify({"error": "could not extract headline"}), 400
    if contains_inappropriate_content(headline):
        return jsonify({"error": "inappropriate content detected"}), 400

    prediction, confidence = predict_raw(headline)
    label = "REAL" if str(prediction) == "1" else "FAKE"
    if prediction == "error":
        return jsonify({"error": "model not loaded"}), 500

    save_prediction(headline, prediction, confidence, request.username)
    return jsonify({
        "url": url,
        "headline": headline,
        "prediction": label, "confidence": round(min(confidence * 100, 99.99), 2)
    })


@app.route("/api/web-predict", methods=["GET"])
@token_required
def api_web_predict():
    all_headlines = fetch_all_headlines_parallel(max_pages=3)
    predictions = []
    for headline in all_headlines:
        if contains_inappropriate_content(headline):
            continue
        prediction, confidence = predict_raw(headline)
        if prediction != "error":
            save_prediction(headline, prediction, confidence, request.username)
            predictions.append({"headline": headline, "prediction": prediction, "confidence": confidence})
    return jsonify({"total_headlines": len(predictions), "predictions": predictions})



def get_all_existing_headlines():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT headline FROM predictions")
        rows = c.fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception as e:
        print(f"⚠️ Bulk DB Check failed: {e}")
        return set()

@app.route("/api/fetch-and-save", methods=["POST"])
@token_required
def api_fetch_and_save():
    # 1. Fetch all headlines (Parallel)
    all_headlines = fetch_all_headlines_parallel(max_pages=3)
    
    # 2. Filter existing (Bulk Check)
    existing_headlines = get_all_existing_headlines()
    new_headlines = [h for h in all_headlines if h not in existing_headlines and not contains_inappropriate_content(h)]
    
    if not new_headlines:
        return jsonify({"success": True, "total_headlines_saved": 0, "message": "No new headlines found."})

    saved_count = 0
    try:
        # 3. Batch Predict (Vectorized - Much Faster)
        # transform all new headlines at once
        embeddings = batch_transform_headlines(new_headlines)
        
        # predict all at once
        if clf:
            probas = clf.predict_proba(embeddings)
            predictions = clf.predict(embeddings)
            
            # 4. Save Loop (Fast DB Inserts)
            conn = get_db_connection()
            c = conn.cursor()
            
            # Prepare batch insert data
            values_to_insert = []
            current_time = datetime.now()
            
            for i, headline in enumerate(new_headlines):
                pred_idx = predictions[i]
                confidence = probas[i][pred_idx]
                label = "REAL" if pred_idx == 1 else "FAKE"
                
                values_to_insert.append(
                    (str(headline), str(label), float(confidence), request.username, current_time)
                )
                saved_count += 1
            
            # Bulk Insert
            c.executemany(
                "INSERT INTO predictions (headline, prediction, confidence, username, timestamp) VALUES (%s, %s, %s, %s, %s)",
                values_to_insert
            )
            conn.commit()
            conn.close()
            
    except Exception as e:
        print(f"❌ Batch Processing Failed: {e}")
        return jsonify({"error": "Batch processing failed"}), 500
            
    return jsonify({"success": True, "total_headlines_saved": saved_count})

# ===== Health route =====
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "model_loaded": clf is not None, "lime_available": LIME_AVAILABLE})


# ===== Background Scheduler =====
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

def headline_exists(headline):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # Check if headline exists (exact match)
        c.execute("SELECT id FROM predictions WHERE headline = %s LIMIT 1", (headline,))
        exists = c.fetchone() is not None
        conn.close()
        return exists
    except Exception as e:
        print(f"⚠️ DB Check failed: {e}")
        return False

def scheduled_news_fetch():
    print(f"\n⏰ [Scheduler] Starting automatic news fetch at {datetime.now()}...")
    try:
        # 1. Fetch from all sources PARALLEL
        all_headlines = fetch_all_headlines_parallel(max_pages=1)
        print(f"📰 [Scheduler] Found {len(all_headlines)} headlines. Analyzing...")

        saved_count = 0
        for headline in all_headlines:
            if contains_inappropriate_content(headline):
                continue
            
            # Check if already exists to avoid duplicate DB entries
            if headline_exists(headline):
                # print(f"⏩ Skipping existing: {headline[:30]}...")
                continue
            
            # 2. Predict
            prediction, confidence = predict_raw(headline)
            if prediction == "error":
                continue

            # 3. Hybrid Logic (Copy of api_predict logic)
            matched_sources, credibility = check_credibility(headline, max_pages=1, threshold=0.70)
            label = "REAL" if str(prediction) == "1" else "FAKE"
            
            HIGH_STAKES_KEYWORDS = ["dead", "died", "killed", "passed away", "murder", "suicide", "blast", "explosion", "attack", "crash", "fatal"]
            is_high_stakes = any(k in headline.lower() for k in HIGH_STAKES_KEYWORDS)
            
            if label == "REAL" and credibility < 20 and is_high_stakes:
                label = "FAKE"
                confidence = 0.99
            
            # 4. Save
            # We use a special username to indicate system auto-fetch
            save_prediction(headline, label, confidence, username="System_Auto")
            saved_count += 1
            
        print(f"✅ [Scheduler] Finished. Saved {saved_count} new predictions.\n")
        
    except Exception as e:
        print(f"❌ [Scheduler] Job failed: {e}")

# Initialize Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_news_fetch, trigger="interval", minutes=30)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())


# ===== Main =====
if __name__ == "__main__":
    # Ensure debug is enabled in development only
    debug_flag = os.environ.get("FLASK_ENV", "development") != "production"
    
    # In debug mode, Flask reloader spawns a child process. 
    # We want the scheduler ONLY in the main process (or the reloader child, but not both).
    # Usually WERKZEUG_RUN_MAIN is 'true' in the reloader child.
    if not debug_flag or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("🚀 Scheduler active.")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug_flag)
