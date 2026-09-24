# Scorecard: ppo_navigate_v3_mix_best.pth (navigate task)

2 episodes per cell, seed 2024.  Cell = success rate / mean event score / grades 0-1-2-3.

| scenario | none | manual | assisted | autonomous |
|---|---|---|---|---|
| head_on | 1.00 / 52 / 0-2-0-0 | 1.00 / 53 / 0-2-0-0 | 1.00 / 53 / 0-2-0-0 | 1.00 / 52 / 0-2-0-0 |
| crossing_starboard | 1.00 / 43 / 1-1-0-0 | 1.00 / 43 / 1-1-0-0 | 1.00 / 42 / 1-1-0-0 | 1.00 / 44 / 1-1-0-0 |
| crossing_port | 0.50 / 59 / 0-1-1-0 | 0.50 / 59 / 0-1-1-0 | 0.50 / 57 / 0-1-1-0 | 0.50 / 58 / 0-1-1-0 |
| overtaking | 1.00 / 51 / 0-2-0-0 | 1.00 / 51 / 0-2-0-0 | 1.00 / 51 / 0-2-0-0 | 1.00 / 21 / 2-0-0-0 |
| overtaken | 1.00 / 44 / 1-1-0-0 | 1.00 / 44 / 1-1-0-0 | 1.00 / 42 / 1-1-0-0 | 1.00 / 25 / 1-1-0-0 |
| data | 1.00 / 32 / 1-1-0-0 | 1.00 / 32 / 1-1-0-0 | 1.00 / 30 / 1-1-0-0 | 1.00 / 29 / 1-1-0-0 |

## Ability index

- label: ppo_navigate_v3_mix_best.pth
- task: navigate
- n_episodes: 48
- success_rate: 0.9166666666666666
- mean_event_score: 44.54791666666667
- robustness_min_success_over_automation: 0.9166666666666666
- hardest_scenario: crossing_port
- hardest_automation: none
- effort_rudder_change_per_step: 0.4141073002900124
- grade_hist: {'safe_passage': 14, 'close_quarters': 30, 'domain_infringement': 4, 'collision': 0}
- efficiency_vs_baseline: 0.9676528134943957
- safety_score: 55.45208333333333
- colreg_compliance: 0.38235294117647056
