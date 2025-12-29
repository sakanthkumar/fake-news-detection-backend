def test_logic(headline, expected_verdict):
    print(f"\nTesting Headline: '{headline}'")
    
    # Mock Inputs
    prediction = "1" # Model says REAL
    credibility = 0  # Not found in trusted sources
    
    label = "REAL" if str(prediction) == "1" else "FAKE"
    
    # --- HYBRID LOGIC (Copied from app.py) ---
    HIGH_STAKES_KEYWORDS = ["dead", "died", "killed", "passed away", "murder", "suicide", "blast", "explosion", "attack", "crash", "fatal"]
    is_high_stakes = any(k in headline.lower() for k in HIGH_STAKES_KEYWORDS)
    
    if label == "REAL" and credibility < 20 and is_high_stakes:
        print(f"⚠️ Hybrid Override: Model said REAL but credibility is {credibility}% for High Stakes headline. Flagging as FAKE.")
        label = "FAKE"
    # -----------------------------------------
    
    print(f"Final Verdict: {label}")
    
    if label == expected_verdict:
        print("✅ PASS")
    else:
        print(f"❌ FAIL (Expected {expected_verdict}, got {label})")

if __name__ == "__main__":
    # Case 1: Hoax (High Stakes) -> Should be FAKE
    test_logic("Virat Kohli has been spotted dead yesterday", "FAKE")
    
    # Case 2: Niche News (Low Stakes) -> Should remain REAL
    test_logic("Local school wins district championship", "REAL")
