# Scorecard: ppo_attack_v3_mix_best.pth (attack task)

2 episodes per cell, seed 2024.  Cell = success rate / mean event score / grades 0-1-2-3.

| scenario | none | manual | assisted | autonomous |
|---|---|---|---|---|
| head_on | 0.00 / 34 / 1-1-0-0 | 0.00 / 34 / 1-1-0-0 | 0.00 / 41 / 1-1-0-0 | 0.00 / 34 / 2-0-0-0 |
| crossing_starboard | 0.50 / 62 / 0-1-1-0 | 0.50 / 64 / 0-1-1-0 | 0.50 / 63 / 0-1-1-0 | 0.00 / 46 / 1-1-0-0 |
| crossing_port | 0.50 / 39 / 1-0-1-0 | 0.00 / 33 / 1-1-0-0 | 0.50 / 43 / 1-0-1-0 | 0.50 / 51 / 1-0-1-0 |
| overtaking | 0.00 / 54 / 0-2-0-0 | 0.00 / 51 / 0-2-0-0 | 0.00 / 52 / 0-2-0-0 | 0.00 / 32 / 1-0-1-0 |
| overtaken | 1.00 / 66 / 0-0-2-0 | 1.00 / 65 / 0-0-2-0 | 0.50 / 58 / 0-1-1-0 | 0.00 / 45 / 1-1-0-0 |
| data | 1.00 / 73 / 0-0-2-0 | 1.00 / 76 / 0-0-2-0 | 1.00 / 76 / 0-0-2-0 | 0.50 / 67 / 0-1-1-0 |

## Ability index

- label: ppo_attack_v3_mix_best.pth
- task: attack
- n_episodes: 48
- success_rate: 0.375
- mean_event_score: 52.52354166666667
- robustness_min_success_over_automation: 0.16666666666666666
- hardest_scenario: head_on
- hardest_automation: autonomous
- effort_rudder_change_per_step: 0.367594559329315
- grade_hist: {'safe_passage': 12, 'close_quarters': 17, 'domain_infringement': 19, 'collision': 0}
- efficiency_vs_baseline: 0.709224380945538
