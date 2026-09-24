# Scorecard: scripted baseline (navigate task)

3 episodes per cell, seed 2024.  Cell = success rate / mean event score / grades 0-1-2-3.

| scenario | none | manual | assisted | autonomous |
|---|---|---|---|---|
| head_on | 0.33 / 46 / 1-2-0-0 | 0.33 / 53 / 1-1-1-0 | 0.33 / 54 / 1-1-1-0 | 0.67 / 54 / 2-0-1-0 |
| crossing_starboard | 0.67 / 52 / 1-2-0-0 | 0.67 / 58 / 1-1-1-0 | 0.33 / 61 / 1-0-2-0 | 1.00 / 44 / 3-0-0-0 |
| crossing_port | 0.67 / 49 / 2-0-1-0 | 0.67 / 50 / 2-0-1-0 | 0.33 / 50 / 1-1-1-0 | 1.00 / 42 / 2-1-0-0 |
| overtaking | 1.00 / 38 / 1-2-0-0 | 1.00 / 38 / 1-2-0-0 | 0.67 / 35 / 2-1-0-0 | 0.67 / 31 / 2-1-0-0 |
| overtaken | 1.00 / 43 / 2-1-0-0 | 1.00 / 43 / 2-1-0-0 | 1.00 / 43 / 2-1-0-0 | 1.00 / 39 / 2-1-0-0 |
| data | 0.67 / 31 / 1-2-0-0 | 0.67 / 39 / 1-2-0-0 | 1.00 / 22 / 2-1-0-0 | 1.00 / 22 / 2-1-0-0 |

## Ability index

- label: scripted baseline
- task: navigate
- n_episodes: 72
- success_rate: 0.736111111111111
- mean_event_score: 43.153333333333336
- robustness_min_success_over_automation: 0.611111111111111
- hardest_scenario: head_on
- hardest_automation: assisted
- effort_rudder_change_per_step: 0.06070091080329465
- grade_hist: {'safe_passage': 38, 'close_quarters': 25, 'domain_infringement': 9, 'collision': 0}
- safety_score: 56.846666666666664
- colreg_compliance: 0.6176470588235294
