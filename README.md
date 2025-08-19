# 🧅 Shallot

A simple, lightweight, unopinionated context management system for [Claude Code](https://www.anthropic.com/claude-code).

## Why?

During long conversations, Claude's context window fills with irrelevant conversation, file contents, and commands, which reduces performance and distracts from current tasks. Meanwhile, important project context like architecture decisions, coding patterns, and folder structure gets lost between conversations. Shallot maintains clean, relevant context that persists across conversations.

## Quick Start

1. Install Shallot in your project directory.

```bash
curl -fsSL https://raw.githubusercontent.com/dylanebert/shallot/main/install.sh | bash
```

2. Initialize the context management system.

```bash
claude
/plant "[optional short description of the project]"
```

**Recommended:** Manually edit `CLAUDE.md` and `layers/structure.md` as needed to fit your project and preferences.

## Usage

1. Use **🧄 `/peel [prompt]`** to load context at the beginning of each conversation
2. Use **🍃 `/nourish`** to update the context and clean up after completing work

## How it Works

Shallot organizes context in four tiers:

1. **Global** (`CLAUDE.md`) - Global standards and system overview
2. **Project** (`layers/structure.md`) - Main project entrypoint
3. **Folder** (`context.md`) - High-level context for each folder
4. **Implementation** (code) - Active work

`/peel` then loads relevant context, while `/nourish` updates context.

## Best Practices

-   Work in small conversations that target a single task each, keeping the context window focused
-   Use `/peel` with a specific prompt at the beginning of each conversation, e.g. `/peel "implement user authentication"`
-   Use `/nourish` at the end of each conversation
-   Use `/clear` after each `/nourish` to clear the conversation
-   Break down larger tasks into smaller conversations, e.g.:

```bash
/peel "outline a character control system. don't implement it yet"
# ...
/nourish
/clear

/peel "begin implementing the character control system. focus for now just on one small, testable iteration"
# ...
/nourish
/clear
```
