# Puzzle Generation Agent

A production-grade **multi-agent system** that synthesizes and validates puzzles
across **25 rule types**. Built on **LangGraph**, powered by the **DeepSeek LLM**,
with a self-correcting generation loop, deterministic structural validation,
two-tier memory, and both conversational and batch interfaces.

The system takes an eval set of 250 reference puzzles (`puzzle.jsonl`, 25 rules ×
10) and generates new, non-duplicate, validated puzzles in the same format.

---

## Highlights

- **Multi-agent pipeline** — Generator → Solver → Reviewer, mutually blind, for
  cross-checked correctness.
- **Self-correction loop** — generation errors are fed back into the prompt; the
  Generator retries up to 5 times.
- **Deterministic validation** — Latin-square / Sudoku / cryptarithm / 24-points
  checks catch LLM mistakes that an LLM judge alone would miss.
- **Tool use (function calling)** — agents can call brute-force solvers and
  validators during generation (`use_tools: true`).
- **Two-tier memory** — STM (Dict / Redis) for session context, LTM
  (SQLite / PostgreSQL) for cross-session analytics.
- **LangGraph Studio** — visualize and step through the pipeline node-by-node.
- **Tracing + diagnosis** — every LLM call is traced, auto-diagnosed, and turned
  into SFT recommendations.
- **Query rewriting** — natural-language intent parsing (regex fast-path + LLM
  fallback), including multi-rule and clarification handling.

---

## Pipeline

The LangGraph graph (`graph.py`):

```
START
  → rewrite_query        # parse natural-language input into structured intent
  → dispatcher           # build per-rule generation jobs
  → llm_synthesizer      # Generator agent (planning + self-correction)
  → llm_crosscheck       # Solver + structural validation
  → verification         # format / dedup / difficulty gate
       ├─ accepted →  data_preprocessor → (loop next item)
       └─ rejected →  llm_synthesizer (retry)
  → summarize            # aggregate run stats (production / quality / tooling)
  → save_output          # write fine_dataset.jsonl + run_report.json
  → END
```

The conversational agent (`agent.py`) wraps the same generation core behind a
multi-agent pipeline (`agents.py`) with intent routing and memory.

---

## Module Map

| Module | Purpose |
|--------|---------|
| `agent.py` | Conversational entry point; intent routing + response shaping |
| `agents.py` | Multi-agent pipeline: Generator / Solver / Reviewer (+ tool calling) |
| `graph.py` | LangGraph pipeline (rewrite → dispatch → synth → verify → save) |
| `graph_v2.py` | Experimental V2: checkpointing + interrupt + per-category subgraphs |
| `llm_gen.py` | DeepSeek generation client and `LlmRule` wrapper |
| `prompt_builder.py` | Layered prompts (system → rule → examples → task) with TODO planning |
| `rewriter.py` | Query intent parser (regex fast-path + LLM fallback, multi-rule) |
| `validators.py` | Per-rule structural validators |
| `tools.py` | Deterministic tools: 24-points / cryptarithm solvers, grid validators |
| `memory.py` | STM (Dict/Redis) + LTM (SQLite/PostgreSQL), unified interface |
| `tracer.py` | Per-call trace + auto-diagnosis + SFT recommendations |
| `session.py` | First-generation vs supplement routing |
| `regression.py` | Prompt versioning + automated regression tests |
| `scheduler.py` | Parallel executor + rate limiter + backoff |
| `summary.py` | Run-summary aggregation (production / quality / tooling / memory) |
| `server.py` | LangGraph Studio entry point |
| `state.py` | LangGraph shared state schema |
| `config.py` | Config loader (YAML + env) |

---

## The 25 Rules

| Category | IDs | Types |
|----------|-----|-------|
| **Word** | 1–8, 24 | Brain Teasers, Affixes, Connect Words, Anagram, Crypto-Math, Word Ladder, Logic, Word Search, Wordscapes |
| **Math** | 9–17, 25 | Math Path, 24 Points, Survo, Kukurasu, Numbrix, Number Wall, Sudoku, Calcudoko, Futoshiki, Skyscrapers |
| **Spatial** | 18–23 | Vector, Star Battle, Campsite, Minesweeper, Arrow Maze, Norinori |

---

## Setup

```bash
pip install -r requirements.txt
```

Provide the API key via environment variable (the config file ships with a blank
key):

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"
```

```bash
# or .env file
echo "DEEPSEEK_API_KEY=sk-your-key-here" > .env
```

---

## Usage

### 1. Conversational mode

```bash
python scripts/chat.py
# or single-shot:
python scripts/chat.py --query "generate 3 puzzles for rule 25"
# with tool calling enabled:
python scripts/chat.py --tools --query "give me 2 sudoku puzzles"
```

Examples it understands:

```
"list all rules"                     show the 25 puzzle types
"show me rule 10"                    inspect rules + examples
"generate 3 puzzles for rule 25"     create 3 Skyscrapers puzzles
"give me 5 easy math puzzles"        domain expansion (math rules)
"rules 1,2,3 each 2 puzzles"         parallel multi-rule generation
"validate the data"                  dedup + structural quality check
"export to data/out.jsonl"           save results
"trace"                              show tracer statistics
```

### 2. Batch mode

```bash
# single rule
python scripts/run.py --rules 4 --count 5

# multiple rules in parallel
python scripts/run.py --rules 4,10,25 --count 2 --workers 4
```

Outputs land in `data/out/`:
- `fine_dataset.jsonl` — accepted puzzles
- `run_report.json` — per-rule counts + timing

### 3. LangGraph Studio

```bash
langgraph dev
```

Open `https://studio.langchain.com`, connect to the local server, and submit
input in the panel:

```json
{"rules": ["4", "10"], "count": 5}
```
or natural language:
```json
{"user_query": "generate 5 puzzles for rule 4"}
```

### 4. Other scripts

```bash
python scripts/calibrate.py       # inspect difficulty calibration per rule
python scripts/verify_output.py   # post-hoc validation of a generated dataset
python scripts/test_tools.py      # exercise the deterministic tools
```

---

## Configuration (`config.yaml`)

```yaml
run:
  rules: ["1", ..., "25"]    # which rules to generate
  count_per_rule: 20
  max_retries_per_item: 15

generator:
  model: deepseek-chat       # or deepseek-reasoner for CoT-heavy rules
  api_key: ""                # leave blank; read from DEEPSEEK_API_KEY
  max_generation_attempts: 5 # self-correction retries
  use_tools: true            # function calling for Generator/Solver/Reviewer

memory:
  stm: dict                  # "redis" in production
  ltm: sqlite                # "postgres" in production
  db_path: data/memory.db
```

---

## Production Deployment

```yaml
# config.yaml overrides
memory:
  stm: redis
  ltm: postgres
  redis_url: "${REDIS_URL}"
  database_url: "${DATABASE_URL}"

generator:
  model: deepseek-reasoner
```

```bash
pip install redis psycopg2-binary
export REDIS_URL=redis://your-redis:6379/0
export DATABASE_URL=postgresql://user:pass@host:5432/puzzle_agent
```

If Redis/PostgreSQL are unavailable, the system automatically falls back to
Dict/SQLite.

---

## How Generation Works

1. **Intent parsing** — the query is decomposed into structured intents
   (rules, count, difficulty, first vs supplement mode).
2. **Prompt construction** — layered prompt with an explicit `<planning>` TODO
   block forcing the model to reason before emitting JSON.
3. **Generator agent** — produces a puzzle; on validation failure the errors are
   appended to the prompt and it retries (up to 5 times).
4. **Solver agent** — independently solves the puzzle (temperature 0, blind to
   the Generator).
5. **Reviewer agent** — compares the two answers, scores 1–10, issues PASS/FAIL.
6. **Structural validation** — deterministic checks (Latin square, cryptarithm,
   expression evaluation, grid shape).
7. **Deduplication** — against the eval set and previously generated puzzles.
8. **Memory** — STM holds session context; LTM persists puzzles + generation log
   for cross-session analytics and SFT mining.

---

## Project Structure

```
.
├── config.yaml                  # main configuration
├── langgraph.json               # LangGraph Studio descriptor
├── puzzle.jsonl                 # eval set (250 puzzles, 25×10)
├── requirements.txt
├── scripts/
│   ├── chat.py                  # conversational CLI
│   ├── run.py                   # batch generation (single + parallel)
│   ├── calibrate.py             # difficulty calibration
│   ├── verify_output.py         # post-hoc validation
│   └── test_tools.py            # tool tests
├── data/
│   └── wordlist.txt             # dictionary for word-based puzzles
└── src/puzzle_agent/
    ├── agent.py / agents.py     # conversational agent + multi-agent pipeline
    ├── graph.py / graph_v2.py   # LangGraph pipelines
    ├── llm_gen.py               # DeepSeek client
    ├── prompt_builder.py        # layered prompt construction
    ├── rewriter.py              # intent parser
    ├── validators.py / tools.py # validation + deterministic solvers
    ├── memory.py                # STM + LTM
    ├── tracer.py                # trace + diagnosis
    ├── session.py               # first/supplement routing
    ├── regression.py            # prompt regression testing
    ├── scheduler.py             # parallel executor + rate limiter
    ├── summary.py               # run-summary aggregation
    ├── server.py / state.py / config.py
    └── rules/                   # rule base classes
```

---

## Design Notes

- **LangGraph** gives a stateful, cyclic pipeline (generate → verify → retry) with
  visual debugging — cleaner than burying control flow in `while`/`try` blocks.
- **Multi-agent** separation (create / solve / judge) provides mutual verification
  that a single LLM call cannot.
- **Self-correction with error feedback** is the single biggest quality lever,
  raising first-pass acceptance substantially over naive one-shot generation.
- **Deterministic validators** are the backstop: an LLM may *claim* a Sudoku is
  valid, but the validator proves it.
- **Two-tier memory** cleanly separates ephemeral conversation context from
  durable analytics used for diagnosis and SFT.

---

## License

MIT
