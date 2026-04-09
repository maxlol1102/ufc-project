from flask import Flask, render_template, jsonify, request
from ai_model.model import get_model
from scraper import get_upcoming_events, get_event_fights

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Load & train model once at startup ──
# Paths are relative to app.py (project root)
_model = None

def model():
    global _model
    if _model is None:
        _model = get_model(
            fights_path="data/fight_data.csv",
            fighters_path="data/ufc_fighters_clean.csv"
        )
    return _model


# ══════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/how_it_works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/bayes_ai")
def bayes_ai():
    return render_template("bayes_ai.html")

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


# ══════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════

@app.route("/api/fighters")
def api_fighters():
    """
    GET /api/fighters
    Returns all fighters grouped by weight class.
    Response:
      {
        "Lightweight":  ["Fighter A", "Fighter B", ...],
        "Welterweight": [...],
        ...
      }
    """
    try:
        data = model().get_fighters_by_division()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fighter_stats")
def api_fighter_stats():
    """
    GET /api/fighter_stats?name=Islam+Makhachev
    Returns stats card for a single fighter.
    """
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name param required"}), 400
    try:
        stats = model().get_fighter_stats(name)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body (JSON): { "fighter_a": "...", "fighter_b": "...", "weight_class": "..." }

    Response:
      {
        winA, winB,
        koA, subA, decA,
        koB, subB, decB,
        koRoundsA, subRoundsA, koRoundsB, subRoundsB,
        skillA, skillB, sigmaA, sigmaB,
        recordA, recordB
      }
    """
    body = request.get_json(force=True) or {}
    fighter_a    = body.get("fighter_a", "").strip()
    fighter_b    = body.get("fighter_b", "").strip()
    weight_class = body.get("weight_class", "").strip() or None

    if not fighter_a or not fighter_b:
        return jsonify({"error": "fighter_a and fighter_b are required"}), 400
    if fighter_a == fighter_b:
        return jsonify({"error": "Fighters must be different"}), 400

    try:
        result = model().predict(fighter_a, fighter_b, weight_class)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500






@app.route("/api/upcoming_events")
def api_upcoming_events():
    """
    GET /api/upcoming_events
    Scrapes ufcstats.com and returns upcoming UFC events.
 
    Response (list):
      [
        { "name": "UFC 314 ...", "url": "http://...", "date": "Apr 12, 2025", "location": "..." },
        ...
      ]
    """
    try:
        events = get_upcoming_events()
        return jsonify(events)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/api/event_fights")
def api_event_fights():
    """
    GET /api/event_fights?url=http://www.ufcstats.com/event-details/...
    Scrapes a single event page and returns its full fight card.
 
    Response:
      {
        "title":   "UFC 314 ...",
        "date":    "April 12, 2025",
        "venue":   "Kaseya Center, Miami",
        "location": "...",
        "fights": [
          {
            "fighter_a":     "Charles Oliveira",
            "fighter_b":     "Michael Chandler",
            "fighter_a_url": "http://...",
            "fighter_b_url": "http://...",
            "weight_class":  "Lightweight",
            "card_section":  "main"
          },
          ...
        ]
      }
    """
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "url param required"}), 400
    # Safety: only allow ufcstats.com URLs
    if "ufcstats.com" not in url:
        return jsonify({"error": "Only ufcstats.com URLs are permitted"}), 400
    try:
        data = get_event_fights(url)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)
    
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)