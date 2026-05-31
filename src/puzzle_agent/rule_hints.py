"""Rule-specific generation hints for complex puzzle types.
Appended to the task prompt when the LLM needs extra guidance.
"""
from typing import Dict

RULE_HINTS: Dict[str, str] = {

    "8": """
### ⚠️ Important for Word Search
Your answer MUST be in this EXACT format:
[[WORD1 (row_start,col_start)(row_end,col_end)
WORD2 (row_start,col_start)(row_end,col_end)]]

- Row numbers start at 1 (leftmost) and column numbers start at 1 (topmost)
- Use line breaks between different words
- The grid must be rectangular with consistent row lengths
- Include the complete letter grid in the question, followed by the word list
""",

    "19": """
### ⚠️ Important for Star Battle
Your answer MUST be in this EXACT format:
[[A(row,col)
B(row,col)
C(row,col)]]

- List ALL region letters in ALPHABETICAL order
- Use newlines to separate different regions
- Each region gets exactly ONE star (row, col)
- Row and column numbers start at 1
""",

    "23": """
### ⚠️ Important for Norinori
Your answer MUST be in this EXACT format:
[[(row1,col1)(row2,col2),(row3,col3)(row4,col4),...]]

- Each pair is ONE domino (2 adjacent cells)
- Dominoes are separated by commas
- List coordinates from top-left to bottom-right
""",

    "11": """
### ⚠️ Important for Survo
Your grid must be rectangular (e.g., 3x4 or 4x4). The answer format:
[[num1 num2 num3, num4 num5 num6, num7 num8 num9]]
""",

    "12": """
### ⚠️ Important for Kukurasu
Use X for empty cells and 1 for filled cells. The answer format:
[[X X 1 X,1 X 1 1,1 1 X 1]]
""",

    "18": """
### ⚠️ Important for Vector Puzzles
Use ← ↑ → ↓ for arrows. The answer format:
[[arrow1 arrow2 arrow3, arrow4 arrow5 arrow6, arrow7 arrow8 arrow9]]
""",

    "21": """
### ⚠️ Important for Minesweeper
Use X for empty cells and A for mines. The answer format:
[[X 2 A 3 X, X A 3 A A, 1 2 3 3 2]]
""",
}


def get_hint(rule_id: str) -> str:
    return RULE_HINTS.get(rule_id, "")
