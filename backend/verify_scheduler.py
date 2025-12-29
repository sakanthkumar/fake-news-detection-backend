import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.abspath("d:/fake news detection/backend"))

# Mocking modules to avoid full app startup/scheduler start
from unittest.mock import MagicMock
import joblib

# Mock database connection
import mysql.connector
mysql.connector.connect = MagicMock()

# Import app but prevent it from starting the scheduler immediately if possible
# Actually, importing app WILL start the scheduler because it's in global scope.
# But that's fine, we just want to test the function.
import app

# Mock dependencies inside app to avoid real network calls if we want a fast test,
# OR we can let it run real network calls to prove it works.
# Let's mock the scraping to save time and avoid network issues, 
# but we want to verify the LOGIC (looping, predicting, saving).

# Mock scraping functions to return 1 fake headline each
app.get_ndtv_headlines = MagicMock(return_value=["Mock NDTV News"])
app.get_bbc_headlines = MagicMock(return_value=["Mock BBC News"])
app.get_toi_headlines = MagicMock(return_value=[])
app.get_thehindu_headlines = MagicMock(return_value=[])
app.get_reuters_headlines = MagicMock(return_value=[])

# Mock Predict
app.predict_raw = MagicMock(return_value=("1", 0.99)) # Real
app.check_credibility = MagicMock(return_value=(["NDTV"], 100)) # Verified
app.save_prediction = MagicMock()

print("🚀 Running scheduled_news_fetch manually...")
try:
    app.scheduled_news_fetch()
    print("✅ Function executed without error.")
    
    # Verify save_prediction was called
    if app.save_prediction.call_count >= 2:
        print(f"✅ save_prediction called {app.save_prediction.call_count} times (Expected >= 2).")
    else:
        print(f"❌ save_prediction called {app.save_prediction.call_count} times.")

except Exception as e:
    print(f"❌ Function failed: {e}")
