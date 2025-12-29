import sys
import os
from unittest.mock import MagicMock

# Add backend to path
sys.path.append(os.path.abspath("d:/fake news detection/backend"))

# Mocking modules
import mysql.connector
mysql.connector.connect = MagicMock()

import app

# Mock dependencies
app.get_ndtv_headlines = MagicMock(return_value=["Duplicate News", "New News"])
app.get_bbc_headlines = MagicMock(return_value=[])
app.get_toi_headlines = MagicMock(return_value=[])
app.get_thehindu_headlines = MagicMock(return_value=[])
app.get_reuters_headlines = MagicMock(return_value=[])

# Mock DB Check
# We want "Duplicate News" to return True (exists), and "New News" to return False (new)
def mock_headline_exists(headline):
    return headline == "Duplicate News"

app.headline_exists = mock_headline_exists
app.predict_raw = MagicMock(return_value=("1", 0.99))
app.check_credibility = MagicMock(return_value=(["NDTV"], 100))
app.save_prediction = MagicMock()

print("🚀 Running scheduled_news_fetch with duplicates...")
app.scheduled_news_fetch()

# Verification
# "Duplicate News" should be skipped. "New News" should be saved.
# So save_prediction should be called exactly ONCE.
if app.save_prediction.call_count == 1:
    args, _ = app.save_prediction.call_args
    saved_headline = args[0]
    if saved_headline == "New News":
        print("✅ SUCCESS: Only 'New News' was saved. Duplicate was skipped.")
    else:
        print(f"❌ FAIL: Wrong headline saved: {saved_headline}")
else:
    print(f"❌ FAIL: save_prediction called {app.save_prediction.call_count} times (Expected 1).")
