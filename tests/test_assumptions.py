from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch
import evaluate
from data import load_example, validate_rows
from elo import Parameters, decay_ratings, event_changes, expected, outcome
from evaluate import Scores, choose_parameters, walk_forward


class ModelingAssumptions(TestCase):
    def setUp(self):
        self.rows = load_example()
        self.event = self.rows[:4]
        self.pre = {r['player_id']: 1500.0 for r in self.event}

    def test_equal_ratings_and_probability_symmetry(self):
        self.assertEqual(expected(1500, 1500), 0.5)
        self.assertAlmostEqual(expected(1800, 1500) + expected(1500, 1800), 1)

    def test_zero_sum_and_snapshot_not_mutated(self):
        self.pre[self.event[0]['player_id']] = 1700
        before = dict(self.pre)
        changes = event_changes(self.event, self.pre, Parameters())
        self.assertAlmostEqual(sum(changes.values()), 0, places=10)
        self.assertEqual(self.pre, before)

    def test_expected_win_gains_less(self):
        pair = self.event[:2]
        normal = event_changes(pair, self.pre, Parameters())[pair[0]['player_id']]
        strong = dict(self.pre)
        strong[pair[0]['player_id']] = 1800
        gain = event_changes(pair, strong, Parameters())[pair[0]['player_id']]
        self.assertLess(gain, normal)

    def test_finisher_beats_cut_and_cut_scores_order(self):
        self.assertEqual(outcome(self.event[1], self.event[2]), 1)
        self.assertEqual(outcome(self.event[2], self.event[3]), 1)

    def test_ties_receive_half_credit(self):
        other = dict(self.event[0], player_id='fictional_5')
        self.assertEqual(outcome(self.event[0], other), 0.5)
        scores = Scores(); scores.add(0.5, 0.7)
        self.assertEqual(scores.summary()['pairwise_accuracy'], 0.5)
        self.assertAlmostEqual(scores.summary()['brier'], 0.04)

    def test_season_decay(self):
        self.assertEqual(decay_ratings({'fictional_1': 1700}, 0.25)['fictional_1'], 1650)

    def test_duplicate_identity_and_invalid_rounds_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(self.rows + [dict(self.rows[0])])
        changed = deepcopy(self.rows); changed[-4]['player_name'] = 'Different Fictional Name'
        with self.assertRaises(ValueError): validate_rows(changed)
        changed = deepcopy(self.rows); changed[0]['rounds_played'] = 1
        with self.assertRaises(ValueError): validate_rows(changed)

    def test_input_order_does_not_change_results(self):
        self.assertEqual(walk_forward(self.rows, Parameters(), [2021, 2022]),
                         walk_forward(list(reversed(self.rows)), Parameters(), [2021, 2022]))

    def capture_snapshots(self, rows):
        snapshots = []
        original = evaluate.score_event
        def capture(event, pre, previous, scores):
            snapshots.append((event[0]['event_id'], dict(pre)))
            original(event, pre, previous, scores)
        with patch('evaluate.score_event', side_effect=capture):
            walk_forward(rows, Parameters(), range(2015, 2023))
        return snapshots

    def test_current_and_future_results_cannot_change_pre_event_ratings(self):
        baseline = self.capture_snapshots(self.rows)
        changed = deepcopy(self.rows)
        for row in changed:
            if row['season'] >= 2021:
                row['total_strokes'] += 100 if row['player_id'] == 'fictional_1' else 0
        altered = self.capture_snapshots(changed)
        # Includes the 2021 event itself: its snapshot must not see its results.
        self.assertEqual(baseline[:7], altered[:7])
        self.assertNotEqual(baseline[-1], altered[-1])

    def test_same_date_events_do_not_see_each_others_results(self):
        first = deepcopy(self.event)
        second = [dict(r, event_id='synth_same_day', event_name='Synthetic Same Day') for r in first]
        snapshots = self.capture_snapshots(first + second)
        self.assertEqual(snapshots[0][1], snapshots[1][1])

    def test_warmup_updates_but_is_not_scored(self):
        result = walk_forward(self.rows, Parameters(), [2021, 2022])
        self.assertEqual(result['elo']['pairs'], 12)
        snapshots = self.capture_snapshots(self.rows)
        self.assertEqual(set(snapshots[0][1].values()), {1500})
        self.assertNotEqual(set(snapshots[1][1].values()), {1500})

    def test_validation_results_cannot_select_parameters(self):
        baseline = choose_parameters(self.rows)
        changed = deepcopy(self.rows)
        for row in changed:
            if row['season'] >= 2021:
                row['total_strokes'] += 100
        self.assertEqual(baseline, choose_parameters(changed))

    def test_no_empty_or_single_player_evaluation(self):
        with self.assertRaises(ValueError): validate_rows([])
        with self.assertRaises(ValueError): event_changes(self.event[:1], self.pre, Parameters())


if __name__ == '__main__':
    main()
