"""
IPL 2026 Data Module
Contains teams, venues, and schedule for IPL 2026
Source: BBC Sport
"""

from datetime import datetime

# IPL 2026 Teams
TEAMS = {
    "Royal Challengers Bengaluru": {"short": "RCB", "city": "Bengaluru"},
    "Sunrisers Hyderabad": {"short": "SRH", "city": "Hyderabad"},
    "Mumbai Indians": {"short": "MI", "city": "Mumbai"},
    "Kolkata Knight Riders": {"short": "KKR", "city": "Kolkata"},
    "Chennai Super Kings": {"short": "CSK", "city": "Chennai"},
    "Punjab Kings": {"short": "PBKS", "city": "Chandigarh"},
    "Delhi Capitals": {"short": "DC", "city": "Delhi"},
    "Rajasthan Royals": {"short": "RR", "city": "Jaipur"},
    "Lucknow Super Giants": {"short": "LSG", "city": "Lucknow"},
    "Gujarat Titans": {"short": "GT", "city": "Ahmedabad"},
}

# IPL 2026 Venues
VENUES = {
    "Bengaluru": "M Chinnaswamy Stadium",
    "Mumbai": "Wankhede Stadium",
    "Kolkata": "Eden Gardens",
    "Chennai": "MA Chidambaram Stadium",
    "Hyderabad": "Rajiv Gandhi International Stadium",
    "Chandigarh": "HPCA Stadium",
    "Delhi": "Arun Jaitley Stadium",
    "Guwahati": "Barsapara Cricket Stadium",
    "Ahmedabad": "Narendra Modi Stadium",
    "Jaipur": "Sawai Mansingh Stadium",
    "Lucknow": "BRSABV Ekana Cricket Stadium",
}

# IPL 2026 Schedule (70 league matches + playoffs) - Source: BBC Sport
IPL_2026_SCHEDULE = [
    # March 2026
    {"date": "2026-03-28", "home": "Royal Challengers Bengaluru", "away": "Sunrisers Hyderabad", "venue": "Bengaluru"},
    {"date": "2026-03-29", "home": "Mumbai Indians", "away": "Kolkata Knight Riders", "venue": "Mumbai"},
    {"date": "2026-03-30", "home": "Rajasthan Royals", "away": "Chennai Super Kings", "venue": "Guwahati"},
    {"date": "2026-03-31", "home": "Punjab Kings", "away": "Gujarat Titans", "venue": "Chandigarh"},
    
    # April 2026
    {"date": "2026-04-01", "home": "Lucknow Super Giants", "away": "Delhi Capitals", "venue": "Lucknow"},
    {"date": "2026-04-02", "home": "Kolkata Knight Riders", "away": "Sunrisers Hyderabad", "venue": "Kolkata"},
    {"date": "2026-04-03", "home": "Chennai Super Kings", "away": "Punjab Kings", "venue": "Chennai"},
    {"date": "2026-04-04", "home": "Delhi Capitals", "away": "Mumbai Indians", "venue": "Delhi"},
    {"date": "2026-04-04", "home": "Gujarat Titans", "away": "Rajasthan Royals", "venue": "Ahmedabad"},
    {"date": "2026-04-05", "home": "Sunrisers Hyderabad", "away": "Lucknow Super Giants", "venue": "Hyderabad"},
    {"date": "2026-04-05", "home": "Royal Challengers Bengaluru", "away": "Chennai Super Kings", "venue": "Bengaluru"},
    {"date": "2026-04-06", "home": "Kolkata Knight Riders", "away": "Punjab Kings", "venue": "Kolkata"},
    {"date": "2026-04-07", "home": "Rajasthan Royals", "away": "Mumbai Indians", "venue": "Guwahati"},
    {"date": "2026-04-08", "home": "Delhi Capitals", "away": "Gujarat Titans", "venue": "Delhi"},
    {"date": "2026-04-09", "home": "Kolkata Knight Riders", "away": "Lucknow Super Giants", "venue": "Kolkata"},
    {"date": "2026-04-10", "home": "Rajasthan Royals", "away": "Royal Challengers Bengaluru", "venue": "Guwahati"},
    {"date": "2026-04-11", "home": "Punjab Kings", "away": "Sunrisers Hyderabad", "venue": "Chandigarh"},
    {"date": "2026-04-11", "home": "Chennai Super Kings", "away": "Delhi Capitals", "venue": "Chennai"},
    {"date": "2026-04-12", "home": "Lucknow Super Giants", "away": "Gujarat Titans", "venue": "Lucknow"},
    {"date": "2026-04-12", "home": "Mumbai Indians", "away": "Royal Challengers Bengaluru", "venue": "Mumbai"},
    {"date": "2026-04-13", "home": "Sunrisers Hyderabad", "away": "Rajasthan Royals", "venue": "Hyderabad"},
    {"date": "2026-04-14", "home": "Chennai Super Kings", "away": "Kolkata Knight Riders", "venue": "Chennai"},
    {"date": "2026-04-15", "home": "Royal Challengers Bengaluru", "away": "Lucknow Super Giants", "venue": "Bengaluru"},
    {"date": "2026-04-16", "home": "Mumbai Indians", "away": "Punjab Kings", "venue": "Mumbai"},
    {"date": "2026-04-17", "home": "Gujarat Titans", "away": "Kolkata Knight Riders", "venue": "Ahmedabad"},
    {"date": "2026-04-18", "home": "Royal Challengers Bengaluru", "away": "Delhi Capitals", "venue": "Bengaluru"},
    {"date": "2026-04-18", "home": "Sunrisers Hyderabad", "away": "Chennai Super Kings", "venue": "Hyderabad"},
    {"date": "2026-04-19", "home": "Kolkata Knight Riders", "away": "Rajasthan Royals", "venue": "Kolkata"},
    {"date": "2026-04-19", "home": "Punjab Kings", "away": "Lucknow Super Giants", "venue": "Chandigarh"},
    {"date": "2026-04-20", "home": "Gujarat Titans", "away": "Mumbai Indians", "venue": "Ahmedabad"},
    {"date": "2026-04-21", "home": "Sunrisers Hyderabad", "away": "Delhi Capitals", "venue": "Hyderabad"},
    {"date": "2026-04-22", "home": "Lucknow Super Giants", "away": "Rajasthan Royals", "venue": "Lucknow"},
    {"date": "2026-04-23", "home": "Mumbai Indians", "away": "Chennai Super Kings", "venue": "Mumbai"},
    {"date": "2026-04-24", "home": "Royal Challengers Bengaluru", "away": "Gujarat Titans", "venue": "Bengaluru"},
    {"date": "2026-04-25", "home": "Delhi Capitals", "away": "Punjab Kings", "venue": "Delhi"},
    {"date": "2026-04-25", "home": "Rajasthan Royals", "away": "Sunrisers Hyderabad", "venue": "Jaipur"},
    {"date": "2026-04-26", "home": "Gujarat Titans", "away": "Chennai Super Kings", "venue": "Ahmedabad"},
    {"date": "2026-04-26", "home": "Lucknow Super Giants", "away": "Kolkata Knight Riders", "venue": "Lucknow"},
    {"date": "2026-04-27", "home": "Delhi Capitals", "away": "Royal Challengers Bengaluru", "venue": "Delhi"},
    {"date": "2026-04-28", "home": "Punjab Kings", "away": "Rajasthan Royals", "venue": "Chandigarh"},
    {"date": "2026-04-29", "home": "Mumbai Indians", "away": "Sunrisers Hyderabad", "venue": "Mumbai"},
    {"date": "2026-04-30", "home": "Gujarat Titans", "away": "Royal Challengers Bengaluru", "venue": "Ahmedabad"},
    
    # May 2026
    {"date": "2026-05-01", "home": "Rajasthan Royals", "away": "Delhi Capitals", "venue": "Jaipur"},
    {"date": "2026-05-02", "home": "Chennai Super Kings", "away": "Mumbai Indians", "venue": "Chennai"},
    {"date": "2026-05-03", "home": "Sunrisers Hyderabad", "away": "Kolkata Knight Riders", "venue": "Hyderabad"},
    {"date": "2026-05-03", "home": "Gujarat Titans", "away": "Punjab Kings", "venue": "Ahmedabad"},
    {"date": "2026-05-04", "home": "Mumbai Indians", "away": "Lucknow Super Giants", "venue": "Mumbai"},
    {"date": "2026-05-05", "home": "Delhi Capitals", "away": "Chennai Super Kings", "venue": "Delhi"},
    {"date": "2026-05-06", "home": "Sunrisers Hyderabad", "away": "Punjab Kings", "venue": "Hyderabad"},
    {"date": "2026-05-07", "home": "Lucknow Super Giants", "away": "Royal Challengers Bengaluru", "venue": "Lucknow"},
    {"date": "2026-05-08", "home": "Delhi Capitals", "away": "Kolkata Knight Riders", "venue": "Delhi"},
    {"date": "2026-05-09", "home": "Rajasthan Royals", "away": "Gujarat Titans", "venue": "Jaipur"},
    {"date": "2026-05-10", "home": "Chennai Super Kings", "away": "Lucknow Super Giants", "venue": "Chennai"},
    {"date": "2026-05-10", "home": "Royal Challengers Bengaluru", "away": "Mumbai Indians", "venue": "Bengaluru"},
    {"date": "2026-05-11", "home": "Punjab Kings", "away": "Delhi Capitals", "venue": "Chandigarh"},
    {"date": "2026-05-12", "home": "Gujarat Titans", "away": "Sunrisers Hyderabad", "venue": "Ahmedabad"},
    {"date": "2026-05-13", "home": "Royal Challengers Bengaluru", "away": "Kolkata Knight Riders", "venue": "Bengaluru"},
    {"date": "2026-05-14", "home": "Punjab Kings", "away": "Mumbai Indians", "venue": "Chandigarh"},
    {"date": "2026-05-15", "home": "Lucknow Super Giants", "away": "Chennai Super Kings", "venue": "Lucknow"},
    {"date": "2026-05-16", "home": "Kolkata Knight Riders", "away": "Gujarat Titans", "venue": "Kolkata"},
    {"date": "2026-05-17", "home": "Punjab Kings", "away": "Royal Challengers Bengaluru", "venue": "Chandigarh"},
    {"date": "2026-05-17", "home": "Delhi Capitals", "away": "Rajasthan Royals", "venue": "Delhi"},
    {"date": "2026-05-18", "home": "Chennai Super Kings", "away": "Sunrisers Hyderabad", "venue": "Chennai"},
    {"date": "2026-05-19", "home": "Rajasthan Royals", "away": "Lucknow Super Giants", "venue": "Jaipur"},
    {"date": "2026-05-20", "home": "Kolkata Knight Riders", "away": "Mumbai Indians", "venue": "Kolkata"},
    {"date": "2026-05-21", "home": "Chennai Super Kings", "away": "Gujarat Titans", "venue": "Chennai"},
    {"date": "2026-05-22", "home": "Sunrisers Hyderabad", "away": "Royal Challengers Bengaluru", "venue": "Hyderabad"},
    {"date": "2026-05-23", "home": "Lucknow Super Giants", "away": "Punjab Kings", "venue": "Lucknow"},
    {"date": "2026-05-24", "home": "Mumbai Indians", "away": "Rajasthan Royals", "venue": "Mumbai"},
    {"date": "2026-05-24", "home": "Kolkata Knight Riders", "away": "Delhi Capitals", "venue": "Kolkata"},
    
    # Playoffs
    {"date": "2026-05-26", "home": "TBD", "away": "TBD", "venue": "TBD", "is_playoff": True, "match_type": "Qualifier 1"},
    {"date": "2026-05-27", "home": "TBD", "away": "TBD", "venue": "TBD", "is_playoff": True, "match_type": "Eliminator"},
    {"date": "2026-05-29", "home": "TBD", "away": "TBD", "venue": "TBD", "is_playoff": True, "match_type": "Qualifier 2"},
    {"date": "2026-05-31", "home": "TBD", "away": "TBD", "venue": "TBD", "is_playoff": True, "match_type": "Final"},
]


def get_team_short(name):
    """Get short name for team."""
    return TEAMS.get(name, {}).get("short", name[:3].upper())


def get_full_team_name(short):
    """Get full name from short name."""
    for name, info in TEAMS.items():
        if info["short"] == short:
            return name
    return short


def get_venue_full_name(city):
    """Get full venue name from city."""
    return VENUES.get(city, city)


def get_upcoming_matches(days_ahead=30):
    """Get upcoming matches within specified days."""
    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    
    matches = []
    for match in IPL_2026_SCHEDULE:
        if match.get("is_playoff"):
            continue
        match_date = datetime.strptime(match["date"], "%Y-%m-%d")
        if today <= match_date <= end_date:
            matches.append(match)
    
    return matches


def get_today_match():
    """Get today's match if any."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    for match in IPL_2026_SCHEDULE:
        if match.get("is_playoff"):
            continue
        if match["date"] == today:
            return match
    
    return None


def get_all_matches():
    """Get all IPL 2026 matches."""
    return [m for m in IPL_2026_SCHEDULE if not m.get("is_playoff")]


def get_playoffs():
    """Get playoff matches."""
    return [m for m in IPL_2026_SCHEDULE if m.get("is_playoff")]


from datetime import timedelta
