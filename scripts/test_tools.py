"""Quick test of tool system integration."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.tools import (
    TOOL_SCHEMAS, get_tool_schemas, get_tool_schemas_for_role,
    get_recommended_tools, ToolExecutor, ToolRegistry, get_tools,
)

print(f"=== Tool Schemas ===")
print(f"Total schemas: {len(TOOL_SCHEMAS)}")
for s in TOOL_SCHEMAS:
    print(f"  - {s['function']['name']}")

print(f"\n=== Role-based Schemas ===")
print(f"Generator: {len(get_tool_schemas_for_role('generator'))}")
print(f"Solver:    {len(get_tool_schemas_for_role('solver'))}")
print(f"Reviewer:  {len(get_tool_schemas_for_role('reviewer'))}")

print(f"\n=== Rule-specific Tools ===")
for rid in ["5", "8", "10", "15", "19", "21", "25"]:
    print(f"  Rule {rid}: {get_recommended_tools(rid)}")

print(f"\n=== Tool Execution ===")
r = get_tools()
print(f"Registry size: {len(r.list_tools())}")

# Test 24points
result = r.call("solve_24points", numbers=[3, 3, 8, 8])
print(f"24points(3,3,8,8): solvable={result['solvable']}, count={result['count']}")

# Test evaluate
result = r.call("evaluate_expression", expression="(6+3)*(5-2)")
print(f"eval((6+3)*(5-2)): {result}")

# Test validate grid
ok, errors = r.call("validate_grid", answer="[[1 2 3, 2 3 1, 3 1 2]]", rule_type="latin_square")
print(f"validate_grid(latin_square): ok={ok}, errors={errors}")

# Test sudoku
result = r.call("solve_sudoku", grid_str="53..7....,6..195...,.98....6.,8...6...3,4..8.3..1,7...2...6,.6....28.,...419..5,....8..79")
print(f"solve_sudoku: solved={result['solved']}, steps={result.get('steps',0)}")

# Test word search
result = r.call("solve_word_search",
    grid=["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXY"],
    words=["ABC", "KLM", "XYZ"])
print(f"word_search: found={result['total_found']}, not_found={result['not_found']}")

# Test minesweeper
result = r.call("solve_minesweeper", grid=["X1X", "2X2", "X1X"])
print(f"minesweeper: mines={result['mines']}, safe={result['safe']}, undetermined={result['undetermined']}")

# Test format converter
result, errors = r.call("format_converter", raw="[[1 2, 3 4]]", target_format="bare")
print(f"format_converter(bare): {result}, errors={errors}")

print(f"\n=== All tests passed ===")
