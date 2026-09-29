"""Educational Elo arithmetic; no stored professional ratings or matchup API."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    K: float = 30
    season_decay: float = 0.25

    def __post_init__(self):
        if self.K <= 0 or not 0 <= self.season_decay <= 1:
            raise ValueError('K must be positive; season decay must be in [0, 1]')


def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def ranking_key(row):
    # A four-round finisher always ranks ahead of a two-round cut.
    return (0 if row['rounds_played'] == 4 else 1, row['total_strokes'])


def outcome(a, b):
    ka, kb = ranking_key(a), ranking_key(b)
    return 1.0 if ka < kb else 0.0 if ka > kb else 0.5


def event_changes(rows, pre_ratings, parameters):
    """Return deltas from a frozen snapshot, without changing the snapshot."""
    if len(rows) < 2:
        raise ValueError('An event needs at least two players')
    changes = {}
    for row in rows:
        pid = row['player_id']
        others = [other for other in rows if other['player_id'] != pid]
        actual = sum(outcome(row, other) for other in others) / len(others)
        predicted = sum(expected(pre_ratings[pid], pre_ratings[other['player_id']]) for other in others) / len(others)
        changes[pid] = parameters.K * (actual - predicted)
    return changes


def decay_ratings(ratings, fraction):
    return {pid: 1500 + (value - 1500) * (1 - fraction) for pid, value in ratings.items()}
