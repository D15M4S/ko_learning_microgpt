# Agent Instructions

## Project Overview
LLM learning project based on @karpathy's microgpt.py for first-time learners.
This project follows the **data flow sequence** to understand how information travels through a GPT model.

## Learning Approach
- **Mode**: Interactive learning with Claude Code
- **Method**: Bottom-up exploration of each component
- **Language**: Korean explanations, English code/comments
- **Base**: microgpt.py (200 lines, dependency-free GPT implementation)

## File Structure
```
ko_review_microgpt/
├── microgpt.py                 # Original reference implementation
├── CLAUDE.md                   # Agent reference pointer
├── AGENT.md                    # This file
├── README.md                   # Detailed documentation (to be created)
│
├── 01_tokenizer/               # String → Token conversion
│   ├── notes.md
│   └── exercises/
│
├── 02_embedding/               # Token/Position embeddings
│   ├── notes.md
│   └── exercises/
│
├── 03_attention/               # Multi-head self-attention
│   ├── notes.md
│   └── exercises/
│
├── 04_mlp/                     # Feed-forward networks
│   ├── notes.md
│   └── exercises/
│
├── 05_output/                  # Linear projection, Softmax
│   ├── notes.md
│   └── exercises/
│
├── 06_loss/                    # Cross-entropy loss calculation
│   ├── notes.md
│   └── exercises/
│
├── 07_backprop/                # Autograd, backward pass
│   ├── notes.md
│   └── exercises/
│
├── 08_optimizer/               # Adam optimizer
│   ├── notes.md
│   └── exercises/
│
└── 09_full_pipeline/           # Complete training & inference
    ├── notes.md
    └── exercises/
```

## Agent Behavior
1. **Ask before implementing** - Clarify requirements first
2. **Explain concepts** - Provide Korean explanations with `✶ Insight` blocks
3. **Request human input** - Use "Learn by Doing" for key implementation decisions
4. **No assumptions** - Ask when uncertain about learning goals
5. **Follow conventions**:
   - Variables/functions: snake_case
   - Classes: PascalCase
   - Docstrings: Google style
   - Comments: Korean allowed for explanations

## Session Workflow
1. User selects a unit (e.g., "03_attention")
2. Agent explains the concept with insights
3. Agent requests human contribution for core logic
4. Human implements the designated section
5. Agent integrates and explains connections to the full pipeline

## Notes
- Each unit's `notes.md` contains: concept explanation, code walkthrough, key insights
- Each unit's `exercises/` contains: practice problems, experiments, variations
- Detailed project goals and learning objectives → see README.md
