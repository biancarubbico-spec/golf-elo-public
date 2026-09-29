"""Load only the bundled fictional teaching fixture."""
import csv
from datetime import date
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / 'examples' / 'synthetic_results.csv'


def validate_rows(rows):
    """Fail rather than guessing how to resolve questionable teaching records."""
    seen = set()
    names, metadata, date_seasons = {}, {}, {}
    for row in rows:
        key = (row['player_id'], row['event_id'])
        if key in seen:
            raise ValueError('Duplicate player-event key')
        seen.add(key)
        if not row['player_id'].startswith('fictional_') or not row['event_name'].startswith('Synthetic '):
            raise ValueError('Only fictional teaching records are supported')
        if row['rounds_played'] not in (2, 4) or row['total_strokes'] <= 0:
            raise ValueError('Use complete two-round cuts or four-round finishes')
        if row['season'] != row['event_date'].year:
            raise ValueError('Fixture uses calendar-year season labels')
        pid = row['player_id']
        if pid in names and names[pid] != row['player_name']:
            raise ValueError('Inconsistent identity')
        names[pid] = row['player_name']
        event = (row['event_date'], row['season'], row['event_name'])
        eid = row['event_id']
        if eid in metadata and metadata[eid] != event:
            raise ValueError('Inconsistent event metadata')
        metadata[eid] = event
        day = row['event_date']
        if day in date_seasons and date_seasons[day] != row['season']:
            raise ValueError('Mixed seasons on one date')
        date_seasons[day] = row['season']
    if not rows:
        raise ValueError('Empty fixture')
    return sorted(rows, key=lambda r: (r['event_date'], r['event_id'], r['player_id']))


def load_example():
    with FIXTURE.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['event_date'] = date.fromisoformat(row['event_date'])
        for key in ('season', 'total_strokes', 'rounds_played'):
            row[key] = int(row[key])
    return validate_rows(rows)
