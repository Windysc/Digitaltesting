# Scorecard: ppo_attack_mix_best.pth (attack task)

3 episodes per cell, seed 2024.  Cell = success rate / mean event score / grades 0-1-2-3.

| scenario | none | manual | assisted | autonomous |
|---|---|---|---|---|
| head_on | 0.00 / 33 / 2-1-0-0 | 0.00 / 33 / 2-1-0-0 | 0.00 / 47 / 1-2-0-0 | 0.00 / 46 / 1-2-0-0 |
| crossing_starboard | 0.33 / 55 / 0-2-1-0 | 0.00 / 51 / 0-3-0-0 | 0.33 / 55 / 0-2-1-0 | 0.33 / 53 / 1-1-1-0 |
| crossing_port | 0.00 / 31 / 2-1-0-0 | 0.00 / 31 / 2-1-0-0 | 0.00 / 47 / 0-3-0-0 | 0.00 / 47 / 1-2-0-0 |
| overtaking | 0.00 / 46 / 1-2-0-0 | 0.33 / 52 / 1-1-1-0 | 0.67 / 56 / 1-0-2-0 | 0.67 / 60 / 0-1-2-0 |
| overtaken | 0.00 / 43 / 2-1-0-0 | 0.00 / 43 / 2-1-0-0 | 0.33 / 48 / 2-0-1-0 | 0.33 / 28 / 2-0-1-0 |
| data | 0.33 / 44 / 1-1-1-0 | 0.33 / 44 / 1-1-1-0 | 0.67 / 51 / 1-0-2-0 | 0.33 / 44 / 1-1-1-0 |

## Ability index

- label: ppo_attack_mix_best.pth
- task: attack
- n_episodes: 72
- success_rate: 0.20833333333333334
- mean_event_score: 45.397777777777776
- robustness_min_success_over_automation: 0.1111111111111111
- hardest_scenario: head_on
- hardest_automation: none
- effort_rudder_change_per_step: 0.06264968785120069
- grade_hist: {'safe_passage': 27, 'close_quarters': 30, 'domain_infringement': 15, 'collision': 0}
- efficiency_vs_baseline: 0.8203571164875715
