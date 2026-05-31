# Puzzle Generation Agent

A production-grade **multi-agent system** for synthesizing and validating puzzles
across 25 rule types. Built with **LangGraph**, powered by **DeepSeek LLM**, featuring
self-correction, structural validation, and conversational interaction.

## Architecture

```
User Query → QueryRewriter (intent parsing)
          → MultiAgentPipeline
             ├── Generator (planning-guided, self-correcting)
             ├── Solver   (independent, temperature=0)
             └── Reviewer (quality scoring, cross-checking)
          → Verification (format + structure + dedup)
          → Memory (STM conversation buffer + LTM persistence)
```

### Key Components

| Layer | Module | Purpose |
|-------|--------|---------|
| **Agent** | `agent.py` | Conversational entry point with intent routing |
| **Multi-Agent** | `agents.py` | Generator → Solver → Reviewer pipeline |
| **Prompt** | `prompt_builder.py` | Layered prompts with TODO-list planning |
| **Validation** | `validators.py` | Structural checks (Latin square, cryptarithm, 24pts) |
| **Tools** | `tools.py` | Deterministic solvers (24pts brute-force, grid validators) |
| **Memory** | `memory.py` | STM (Redis/Dict) + LTM (PostgreSQL/SQLite) |
| **Tracing** | `tracer.py` | Per-call trace + auto-diagnosis + SFT recommendations |
| **Rewrite** | `rewriter.py` | LLM + regex hybrid intent parsing |
| **Session** | `session.py` | First-generation vs supplement routing |
| **Regression** | `regression.py` | Prompt versioning + automated regression tests |
| **Scheduler** | `scheduler.py` | Parallel generation + rate limiter |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Set API Key

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"
```

### Run (Conversational Mode)

```bash
python scripts/chat.py
```

```
🤖 PuzzleAgent — try these:
  "list all rules"                    — show 25 puzzle types
  "show me rule 10"                   — inspect 24-points rules and examples
  "generate 3 puzzles for rule 25"    — create 3 Skyscrapers puzzles
  "give me 5 easy math puzzles"       — domain expansion (rules 9-17,25)
  "rules 1,2,3 each 2 puzzles"        — parallel multi-rule generation
  "validate the data"                 — check dedup + structure quality
  "export to data/out.jsonl"          — save results
  "stats"                             — show generation statistics
```

### Run (Batch Mode)

```bash
python scripts/run.py --rules 1,5,10,25 --count 20
```

### Run (LangGraph Studio)

```bash
langgraph dev
```

Open `https://studio.langchain.com` → connect to local server → visualize the pipeline.

## Configuration

Edit `config.yaml`:

```yaml
run:
  rules: ["all"]           # "all" = all 25 rules, or ["1","5","10"]
  count_per_rule: 20       # puzzles to generate per rule
  max_retries_per_item: 15 # retry budget

generator:
  model: deepseek-chat           # also: deepseek-reasoner for CoT
  max_generation_attempts: 5     # self-correction retries
  max_output_tokens: 4096

memory:
  stm: dict                      # "redis" for production
  ltm: sqlite                    # "postgres" for production
  # redis_url: "redis://..."     # set for production
  # database_url: "postgresql://..."

teacher:
  enabled: false                 # set true for reasoning traces
```

## Production Deployment

```yaml
# config.yaml — production overrides
memory:
  stm: redis
  ltm: postgres
  redis_url: "${REDIS_URL}"
  database_url: "${DATABASE_URL}"

generator:
  model: deepseek-reasoner
  max_generation_attempts: 5
```

```bash
pip install redis psycopg2-binary
export REDIS_URL=redis://your-redis:6379/0
export DATABASE_URL=postgresql://user:pass@host:5432/puzzle_agent
```

## Project Structure

```
.
├── config.yaml                  # Main configuration
├── langgraph.json               # LangGraph Studio descriptor
├── puzzle.jsonl                 # Original eval set (250 puzzles, 25×10)
├── requirements.txt
├── scripts/
│   ├── chat.py                  # Conversational CLI
│   ├── run.py                   # Batch generation
│   ├── calibrate.py             # Difficulty calibration
│   └── verify_output.py         # Post-hoc validation
├── data/
│   └── wordlist.txt             # Dictionary for word-based puzzles
└── src/puzzle_agent/
    ├── agent.py                 # Main conversational agent
    ├── agents.py                # Multi-agent pipeline (Generator/Solver/Reviewer)
    ├── config.py                # Configuration loader
    ├── dedup.py                 # Deduplication utilities
    ├── graph.py                 # LangGraph pipeline
    ├── graph_v2.py              # LangGraph V2 (checkpoint + subgraph)
    ├── llm_gen.py               # DeepSeek generation client
    ├── memory.py                # STM + LTM memory system
    ├── prompt_builder.py        # Layered prompt construction
    ├── regression.py            # Prompt regression testing
    ├── rewriter.py              # Query intent parser
    ├── rule_hints.py            # Per-rule format hints
    ├── scheduler.py             # Parallel executor + rate limiter
    ├── server.py                # LangGraph Studio entry point
    ├── session.py               # Session manager (first/supplement)
    ├── state.py                 # LangGraph shared state
    ├── tools.py                 # Deterministic validation tools
    ├── tracer.py                # Trace + diagnosis system
    ├── validators.py            # Per-rule structural validators
    └── rules/                   # Rule base classes
```

## The 25 Puzzle Rules

| Category | IDs | Types |
|----------|-----|-------|
| **Word** | 1-8, 24 | Brain Teasers, Affixes, Connect Words, Anagram, Crypto-Math, Word Ladder, Logic, Word Search, Wordscapes |
| **Math** | 9-17, 25 | Math Path, 24 Points, Survo, Kukurasu, Numbrix, Number Wall, Sudoku, Calcudoko, Futoshiki, Skyscrapers |
| **Spatial** | 18-23 | Vector, Star Battle, Campsite, Minesweeper, Arrow Maze, Norinori |

## How Generation Works

1. **Intent Parsing** — user query is decomposed into structured intents
2. **Prompt Construction** — layered prompt: system role → rule content → dynamic examples → task
3. **Generator Agent** — creates puzzle with TODO-list planning, self-corrects up to 5 times
4. **Solver Agent** — independently solves the puzzle (temperature=0, blind to generator)
5. **Reviewer Agent** — compares answers, scores 1-10, issues PASS/FAIL verdict
6. **Validation** — structural checks (Latin square, cryptarithm, expression evaluation)
7. **Deduplication** — checks against original eval set and previously generated puzzles
8. **Memory Persistence** — STM for session context, LTM for cross-session analytics

## Key Design Decisions

- **LangGraph** for stateful pipeline orchestration with visual debugging
- **Multi-Agent** architecture (Generate → Solve → Review) for mutual verification
- **Self-correction** loop with error feedback reduces failure rate from ~60% to >90%
- **Structured planning** (`<planning>` blocks) forces the LLM to reason before generating
- **Deterministic validators** catch LLM errors (grid structure, arithmetic, cryptarithm logic)
- **Two-tier memory** separates conversation context (STM) from persistent analytics (LTM)
- **Production-ready** with Redis/PostgreSQL support, rate limiting, parallel execution

## Testing

```bash
# Unit tests for solvers and validators
python tests/test_solvers.py

# Full regression suite (38 tests across 25 rules)
python -c "from src.puzzle_agent.regression import build_standard_suite; build_standard_suite()"

# Post-hoc validation of generated dataset
python scripts/verify_output.py
```

## License

MIT
