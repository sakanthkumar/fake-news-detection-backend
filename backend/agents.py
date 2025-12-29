# agents.py  (offline version)
from colorama import Fore, Style, init
import time, re, requests
from bs4 import BeautifulSoup

# Initialize colorama for Windows
init(autoreset=True)

def log_agent_start(agent_name, message, color=Fore.CYAN):
    print(color + f"\n🔹 [{agent_name}] started...")
    print(Fore.WHITE + f"   └─ Task: {message}")
    time.sleep(0.2)

def log_agent_done(agent_name, color=Fore.GREEN):
    print(color + f"✅ [{agent_name}] completed.\n" + Style.RESET_ALL)
    time.sleep(0.1)

# --- MockAgent replaced by structured offline agents ---

class ScraperAgent:
    role = "scraper"
    name = "Scraper Agent"
    goal = "Scrape web for headline and return sources + credibility score"

    def run(self, input=None):
        headline = input or ""
        log_agent_start(self.name, f"Searching web for: {headline}")
        res = self._scrape_search(headline)
        log_agent_done(self.name)
        return res

    def _scrape_search(self, headline, max_links=6):
        try:
            # Use Google search results page scraping (simple approach)
            q = "+".join(re.findall(r"\w+", headline))
            url = f"https://www.google.com/search?q={q}"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")

            raw_links = []
            for a in soup.select("a"):
                href = a.get("href", "")
                if href.startswith("/url?q="):
                    link = href.split("&")[0].replace("/url?q=", "")
                    raw_links.append(link)
            # dedupe and limit
            cleaned = []
            for u in raw_links:
                if u not in cleaned and len(cleaned) < max_links:
                    cleaned.append(u)

            # compute credibility: how many are from trusted news domains
            trusted = ["bbc.com", "reuters.com", "thehindu.com", "ndtv.com", "cnn.com", "indiatimes.com"]
            hits = sum(1 for u in cleaned if any(d in u for d in trusted))
            cred_score = round((hits / max(1, len(cleaned))) * 100, 2)

            return {
                "headline": headline,
                "sources": cleaned,
                "total_sources_found": len(cleaned),
                "credibility_score": cred_score
            }
        except Exception as e:
            return {"error": f"scrape_failed: {e}"}


class PredictorAgent:
    role = "predictor"
    name = "Predictor Agent"
    goal = "Run the local classifier (BERT+sklearn) to predict fake/real"

    def run(self, input=None):
        log_agent_start(self.name, f"Predicting: {input}")
        # This agent only logs; real prediction uses your predict_raw() in app.py
        time.sleep(0.2)
        log_agent_done(self.name)
        return {"info": "prediction invoked"}


class ExplainerAgent:
    role = "explainer"
    name = "Explainer Agent"
    goal = "Produce a human-friendly explanation from LIME output"

    def run(self, input=None):
        log_agent_start(self.name, f"Explaining: {input}")
        time.sleep(0.2)
        log_agent_done(self.name)
        # The actual explanation production will be done in app.py using lime_explain().
        return {"info": "explainer invoked"}


# instantiate agents for import
scraper_agent = ScraperAgent()
predictor_agent = PredictorAgent()
explainer_agent = ExplainerAgent()
