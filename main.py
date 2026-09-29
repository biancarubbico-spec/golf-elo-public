"""Run the fictional example only. Never export player-level model state."""
import json
from data import load_example
from evaluate import choose_parameters, walk_forward


def main():
    rows = load_example()
    params, tuning = choose_parameters(rows)
    validation = walk_forward(rows, params, (2021, 2022))
    print('SYNTHETIC EDUCATIONAL DEMO: these are not the historical results.')
    print(json.dumps({'dataset': 'bundled fictional fixture',
                     'parameters': {'K': params.K, 'season_decay': params.season_decay},
                     'synthetic_tuning': tuning, 'synthetic_validation': validation}, indent=2))


if __name__ == '__main__':
    main()
