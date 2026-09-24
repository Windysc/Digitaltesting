# Scorecard: scripted baseline (attack task)

3 episodes per cell, seed 2024.  Cell = success rate / mean event score / grades 0-1-2-3.

| scenario | none | manual | assisted | autonomous |
|---|---|---|---|---|
| head_on | 1.00 / 64 / 0-0-3-0 | 1.00 / 72 / 0-0-3-0 | 1.00 / 71 / 0-0-3-0 | 1.00 / 74 / 0-0-3-0 |
| crossing_starboard | 1.00 / 67 / 0-0-3-0 | 1.00 / 65 / 0-0-3-0 | 1.00 / 63 / 0-0-3-0 | 1.00 / 65 / 0-0-3-0 |
| crossing_port | 1.00 / 55 / 0-0-3-0 | 1.00 / 55 / 0-0-3-0 | 1.00 / 56 / 0-0-3-0 | 1.00 / 67 / 0-0-3-0 |
| overtaking | 1.00 / 66 / 0-0-3-0 | 1.00 / 68 / 0-0-3-0 | 1.00 / 68 / 0-0-3-0 | 1.00 / 68 / 0-0-3-0 |
| overtaken | 0.33 / 52 / 2-0-1-0 | 0.33 / 52 / 2-0-1-0 | 0.33 / 48 / 2-0-1-0 | 0.33 / 26 / 2-0-1-0 |
| data | 1.00 / 77 / 0-0-3-0 | 1.00 / 76 / 0-0-3-0 | 1.00 / 72 / 0-0-3-0 | 1.00 / 71 / 0-0-3-0 |

## Ability index

- label: scripted baseline
- task: attack
- n_episodes: 72
- success_rate: 0.888888888888889
- mean_event_score: 63.157777777777774
- robustness_min_success_over_automation: 0.8888888888888888
- hardest_scenario: overtaken
- hardest_automation: none
- effort_rudder_change_per_step: 0.041978106167243255
- grade_hist: {'safe_passage': 8, 'close_quarters': 0, 'domain_infringement': 64, 'collision': 0}
