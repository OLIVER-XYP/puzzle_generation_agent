# Fuzzy Semantic Probe Report

This offline benchmark evaluates whether PuzzleAgent can resolve vague, context-dependent user language into executable agent intents without calling an LLM.

## Summary

- Total probes: 10
- Passed: 10
- Pass rate: 100%
- Generated at: 2026-06-08T20:34:10
- Reproduction: `python scripts/semantic_probe.py --out data/evolution/semantic_probe_report.json`

## Probe Categories

- Context ellipsis: follow-up requests like "two more like the previous one".
- Domain fuzziness: broad category requests like "hard number-grid puzzles".
- Analogy and negation: requests like "similar to Sudoku, but not Sudoku itself".
- Active recommendation: next-best-rule selection from user memory.
- Compound distribution: multi-rule requests using all/each semantics.
- Negative domain constraints: requests such as "not math".
- Cross-lingual fuzziness: mixed English requests such as hard grid logic.

## Results

| Probe | Passed | Actions | Rules | Errors |
|---|---:|---|---|---|
| followup_same_rule | yes | GENERATE | 10 |  |
| domain_fuzzy_math_hard | yes | GENERATE, GENERATE, GENERATE | 11, 12, 13 |  |
| similar_sudoku_not_same | yes | GENERATE, GENERATE, GENERATE | 16, 17, 25 |  |
| recommend_next | yes | RECOMMEND |  |  |
| multi_rule_each | yes | GENERATE, GENERATE | 10, 25 |  |
| word_not_math | yes | GENERATE, GENERATE, GENERATE | 1, 2, 3 |  |
| followup_all_recent_rules | yes | GENERATE, GENERATE | 10, 25 |  |
| analog_24_not_24 | yes | GENERATE, GENERATE, GENERATE | 9, 11, 12 |  |
| negative_spatial_math | yes | GENERATE, GENERATE, GENERATE | 7, 15, 17 |  |
| english_fuzzy_grid | yes | GENERATE, GENERATE, GENERATE | 11, 12, 13 |  |

## Interpretation

A passing run means the agent can deterministically map representative ambiguous requests to stable actions before invoking generation. This is useful for paper/project explanation because it separates semantic routing quality from downstream LLM puzzle-generation quality.

## Paper-Style Claim

The probe suite operationalizes semantic robustness as exact-match intent routing under vague language. The measured pass rate is not a puzzle quality score; it is an upstream controller score showing whether the agent can choose the correct rule family, count, difficulty bucket, and recommendation action before generation.
