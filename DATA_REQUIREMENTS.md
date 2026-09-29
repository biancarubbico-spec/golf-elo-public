# Data provenance and educational schema

The original historical study used [PGA Tour Golf Data (2015–2022) on Kaggle](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022). The source archive contains a tournament-level results CSV. To obtain it independently, open the dataset page, review its description and applicable terms, sign in if required, and use Kaggle's download control. Keep downloaded data outside this public folder. Do not commit it. Dataset terms and availability are controlled by the provider; this edition does not redistribute the source or grant rights to it.

The historical input fields were `player id`, `player`, `tournament id`, `date`, `season`, `strokes`, `n_rounds`, `Finish`, and `tournament name`. One row represents a player's tournament total, not a round or a hole. `date` was an explicit month/day/two-digit-year field. No individual round or explicit holes-played fields were available. Scores for duplicate player–event rows must not be added together. PGA Tour season labels can span two calendar years.

Historical research requires auditing event formats, withdrawals, cut rules, duplicate scores and identities, and date/season conflicts before processing. The complete historical preparation code and records are intentionally private. The public implementation accepts no historical-data command-line argument or holdout import, and does not claim exact historical reproduction.

## Included fictional fixture

Only `examples/synthetic_results.csv` is intended for the runnable example. It contains four invented players in eight invented events, one event in each season from 2015 through 2022. All event names begin with `Synthetic`; all player IDs begin with `fictional_`.

| Field | Meaning |
| --- | --- |
| `player_id` | Stable fictional identifier |
| `player_name` | Fictional display name |
| `event_id` | Fictional tournament identifier |
| `event_name` | Explicitly synthetic event name |
| `event_date` | Invented ISO end date |
| `season` | Invented season label; fixtures use calendar-year dates |
| `total_strokes` | Already aggregated tournament strokes |
| `rounds_played` | Two for a normal cut, four for a completed event |

The public loader rejects duplicate player–event keys, inconsistent names or event metadata, nonpositive scores, invalid rounds, mixed seasons on one date, and season labels inconsistent with its calendar-year fixture. It sorts dates regardless of input row order. These deliberately simplified fixture rules are not PGA Tour season-window validation. Three-round cuts, withdrawals, DQs, team events, shortened events, and other formats are not supported by this loader.
