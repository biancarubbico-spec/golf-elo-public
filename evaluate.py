"""Chronological evaluation; return aggregate metrics only."""
from itertools import combinations, groupby
from math import log
from statistics import mean
from data import validate_rows
from elo import Parameters, decay_ratings, event_changes, expected, outcome


class Scores:
    def __init__(self):
        self.count = 0
        self.loss = self.brier = self.correct = 0.0

    def add(self, y, p):
        clipped = min(max(p, 1e-12), 1 - 1e-12)
        self.loss += -y * log(clipped) - (1-y) * log(1-clipped)
        self.brier += (p-y) ** 2
        self.correct += 0.5 if p == 0.5 else y if p > 0.5 else 1-y
        self.count += 1

    def summary(self):
        if not self.count:
            raise ValueError('No scored comparisons')
        return {'log_loss': self.loss / self.count, 'brier': self.brier / self.count,
                'pairwise_accuracy': self.correct / self.count, 'pairs': self.count}


def score_event(rows, pre_ratings, previous_finishes, scores):
    """Evaluate the frozen snapshot before the caller applies this event's deltas."""
    for a, b in combinations(rows, 2):
        y = outcome(a, b)
        scores['elo'].add(y, expected(pre_ratings[a['player_id']], pre_ratings[b['player_id']]))
        scores['coin_flip'].add(y, 0.5)
        fa = previous_finishes.get(a['player_id'])
        fb = previous_finishes.get(b['player_id'])
        p = 0.5 if fa is None or fb is None or fa == fb else 0.75 if fa < fb else 0.25
        scores['previous_season_finish'].add(y, p)


def walk_forward(rows, parameters, scored_seasons):
    rows = validate_rows(rows)
    ratings, finishes, previous = {}, {}, {}
    current_season = None
    scores = {name: Scores() for name in ('elo', 'coin_flip', 'previous_season_finish')}
    for _, date_rows in groupby(rows, key=lambda r: r['event_date']):
        batch = list(date_rows)
        season = batch[0]['season']
        if season != current_season:
            previous = {p: mean(values) for p, values in finishes.items()} if current_season is not None and season == current_season+1 else {}
            finishes = {}
            if current_season is not None:
                ratings = decay_ratings(ratings, parameters.season_decay)
            current_season = season
        pending = []
        for _, event_rows in groupby(batch, key=lambda r: r['event_id']):
            event = list(event_rows)
            pre = {r['player_id']: ratings.get(r['player_id'], 1500.0) for r in event}
            if season in scored_seasons:
                score_event(event, pre, previous, scores)
            pending.append((event, event_changes(event, pre, parameters)))
        # Same-date snapshots are all frozen before any update.
        for event, changes in pending:
            for row in event:
                pid = row['player_id']
                ratings[pid] = ratings.get(pid, 1500.0) + changes[pid]
                rank = 1 + sum(outcome(other, row) for other in event if other['player_id'] != pid)
                finishes.setdefault(pid, []).append(rank)
    return {name: values.summary() for name, values in scores.items()}


def choose_parameters(rows):
    # Validation rows are removed before any candidate is evaluated.
    tuning_rows = [r for r in rows if r['season'] <= 2020]
    candidates = []
    for k in (10, 20, 30, 40, 60):
        for decay in (0, 0.15, 0.25, 0.4):
            params = Parameters(k, decay)
            result = walk_forward(tuning_rows, params, range(2017, 2021))
            candidates.append((result['elo']['log_loss'], params, result))
    _, params, result = min(candidates, key=lambda candidate: candidate[0])
    return params, result
