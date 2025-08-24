# AI Context - Working Agreement

<project-description>
[Short high-level description of the project - MUST be filled during /plant initialization]
</project-description>

3D game engine layer for vibe coding - ECS-based abstraction for easily building Three.js games using Entity Component System (ECS) architecture with bitECS.

**Required**: Read [layers/structure.md](layers/structure.md) before proceeding with any task

## Context Management System

- **Tier 0 — global**: `CLAUDE.md` (root). Global standards and system overview
- **Tier 1 — project**: `layers/structure.md`. Project map (stack, commands, layout, entry points)
- **Tier 2 — folder context**: `context.md` in any folder; one per folder; explains purpose/structure of that folder
- **Tier 3 — implementation**: Code files (scripts)

## Rules

- **Priority**: Your number one priority is to manage your own context; always load appropriate context before doing anything else
- **No History**: CRITICAL - Code and context must NEVER reference their own history. Write everything as the current, final state. Never include comments like "changed from X to Y" or "previously was Z". This is a severe form of context rot
- **Simplicity**: Keep code simple, elegant, concise, and readable
- **Structure**: Keep files small and single-responsibility; separate concerns (MVC/ECS as appropriate)
- **Reuse**: Reuse before adding new code; avoid repetition
- **Comments**: Code should be self-explanatory without comments; use concise comments only when necessary
- **State**: Single source of truth; caches/derivations only
- **Data**: Favor data-driven/declarative design
- **Fail Fast**: Make bugs immediately visible rather than hiding them; favor simplicity over defensive patterns
- **Backwards Compatibility**: Unless stated otherwise, favor simplicity over backwards compatibility; the design rules above should make breaking changes easy to trace and fix

## Security

- **Inputs & secrets**: Validate inputs; secrets only in env; never log sensitive data
- **Auth**: Gateway auth; server-side token validation; sanitize inputs

## Tools

- **Context7**: Use as needed to fetch documentation
