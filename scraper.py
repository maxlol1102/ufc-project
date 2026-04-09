# ════════════════════════════════════════════════════════
# scraper.py  —  place this in your project root
# Scrapes upcoming events + fight cards from ufcstats.com
# ════════════════════════════════════════════════════════

import requests
from bs4 import BeautifulSoup

BASE    = "http://www.ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFC-Research-Bot/1.0)"}

# ── Weight-class slug normalisation (ufcstats uses raw strings)
WEIGHT_CLASS_CLEAN = {
    "Lightweight":          "Lightweight",
    "Welterweight":         "Welterweight",
    "Middleweight":         "Middleweight",
    "Featherweight":        "Featherweight",
    "Bantamweight":         "Bantamweight",
    "Heavyweight":          "Heavyweight",
    "Light Heavyweight":    "Light Heavyweight",
    "Flyweight":            "Flyweight",
    "Women's Strawweight":  "Women's Strawweight",
    "Women's Flyweight":    "Women's Flyweight",
    "Women's Bantamweight": "Women's Bantamweight",
    "Women's Featherweight":"Women's Featherweight",
    "Catch Weight":         "Catch Weight",
    "Open Weight":          "Open Weight",
}


def get_upcoming_events() -> list[dict]:
    """
    Scrape the ufcstats upcoming-events page.

    Returns a list of dicts:
        [{"name": str, "url": str, "date": str, "location": str}, ...]
    sorted chronologically (nearest first).
    """
    url  = f"{BASE}/statistics/events/upcoming"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    rows   = soup.select("tr.b-statistics__table-row")

    for row in rows:
        link = row.select_one("a.b-link")
        if not link:
            continue

        # Date sits in a <span> inside the first <td>
        date_span = row.select_one("span.b-statistics__date")
        date_str  = date_span.get_text(strip=True) if date_span else ""

        # Location is in the second <td>
        tds      = row.select("td")
        location = tds[1].get_text(strip=True) if len(tds) > 1 else ""

        events.append({
            "name":     link.get_text(strip=True),
            "url":      link["href"],
            "date":     date_str,
            "location": location,
        })

    return events


def get_event_fights(event_url: str) -> dict:
    """
    Scrape a single event page on ufcstats.

    Returns:
        {
          "title":    str,
          "date":     str,
          "venue":    str,
          "location": str,
          "fights": [
              {
                "fighter_a":     str,
                "fighter_b":     str,
                "fighter_a_url": str,
                "fighter_b_url": str,
                "weight_class":  str,
                "card_section":  "main" | "prelim" | "early_prelim",
              },
              ...
          ]
        }

    NOTE: ufcstats does not reliably expose card-section metadata in the
    HTML for upcoming events — we default everything to "main" and rely
    on fight order (first fight listed = main event).  You can override
    this by maintaining a side-list of prelim fights per event.
    """
    resp = requests.get(event_url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Event meta ──────────────────────────────────────────
    title_el = soup.select_one("span.b-content__title-highlight")
    title    = title_el.get_text(strip=True) if title_el else ""

    meta_items = soup.select("li.b-list__box-list-item")
    def meta(label: str) -> str:
        for li in meta_items:
            txt = li.get_text(strip=True)
            if label.lower() in txt.lower():
                return txt.split(":", 1)[-1].strip()
        return ""

    date     = meta("Date")
    location = meta("Location")
    venue    = location  # ufcstats combines venue + city into "Location"

    # ── Fights ──────────────────────────────────────────────
    fights = []
    rows   = soup.select("tr.b-fight-details__table-row[data-link]")

    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 7:
            continue

        # Fighter names + profile URLs
        fighter_links = cols[1].select("a.b-link")
        if len(fighter_links) < 2:
            continue

        name_a = fighter_links[0].get_text(strip=True)
        name_b = fighter_links[1].get_text(strip=True)
        url_a  = fighter_links[0].get("href", "")
        url_b  = fighter_links[1].get("href", "")

        # Weight class — col index 6
        weight_raw   = cols[6].get_text(strip=True) if len(cols) > 6 else ""
        # Strip "Bout" suffix that ufcstats sometimes appends
        weight_clean = weight_raw.replace(" Bout", "").strip()
        weight_class = WEIGHT_CLASS_CLEAN.get(weight_clean, weight_clean)

        fights.append({
            "fighter_a":     name_a,
            "fighter_b":     name_b,
            "fighter_a_url": url_a,
            "fighter_b_url": url_b,
            "weight_class":  weight_class,
            "card_section":  "main",   # ufcstats doesn't expose this clearly
        })

    return {
        "title":    title,
        "date":     date,
        "venue":    venue,
        "location": location,
        "fights":   fights,
    }
