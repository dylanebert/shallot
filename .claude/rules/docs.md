# Docs

Doc pages optionally map to a code folder via `source` frontmatter (shows source link + path). Omit `source` for pages that don't correspond to a specific module (e.g. quick-start). Code is the source of truth. Make the code elegant first, then document what's non-obvious.

`docs/` is for game developers using Shallot. `.claude/rules/` is for engine developers modifying Shallot.

## Structure

- **Start** (`docs/guide/`) — progressive disclosure, builds concepts step by step
- **Engine** (`docs/engine/`) — ECS, scenes, app, utils
- **Standard** (`docs/standard/`) — built-in subsystems
- **Editor** (`docs/editor/`) — editor-enabling engine libraries (document model, session sync) and Svelte editor integration
- **Extras** (`docs/extras/`) — optional plugins

Reference tabs use `<!-- API:path -->` for public API tables and `<!-- CORE:subsystem -->` for core API tables. Both are replaced by `bun run build`. UI and Code tabs are manually written. Reference tabs contain only markers.

## Tabs

Tab groups are atomic blocks wrapped in `<!-- tabs -->` / `<!-- /tabs -->`. Inside, use `<!-- tab: Name -->` for each tab. Tab groups can appear anywhere in the page, multiple times. Content outside tab groups is shared across all tabs. Not every page needs all three tabs — skip tabs that would be empty.

Inline tabs use `<!-- pick -->` / `<!-- /pick -->` with `<!-- pick: Name -->` for each option. Used for scene/code alternatives and OS-specific instructions. OS tabs (`<!-- os -->`) also work — both render the same way. If tab names match an OS ("Windows", "Mac/Linux"), the visitor's OS is auto-detected.

Title and description render from frontmatter above all content, not from the markdown h1.

### UI

Audience: someone who doesn't write code. Should not feel redundant for someone who does.

- Editor workflows only. No TypeScript
- Accessible but not condescending. Don't simplify concepts, simplify the interface to them
- A reader should be able to follow only UI tabs and understand simple usage
- Can point to the Code tab for more customization
- For topics where the editor isn't relevant (e.g. Compute), keep a short conceptual explanation and point to Code. Don't force UI content where it doesn't fit

### Code

Audience: game developers. The primary tab for most topics.

- TypeScript API, scene files, systems, plugins
- Can cover the same flows as UI but with scene files and code
- Extends beyond what the editor can do — custom systems, novel gameplay behavior
- Extension/contributor prose (architecture, buffer layouts, registry internals) goes at the bottom when relevant

### Reference

Auto-generated from JSDoc. No hand-written prose. `bun run build` replaces markers with HTML.

- `<!-- API:path -->` for public exports, `<!-- CORE:subsystem -->` for core API (ecs, compute, render, physics, audio, transforms, raytracing)
- Entries render as a scannable list. `@expand` classes/interfaces show members inline

**JSDoc conventions** (on definition site, not barrel):

- lowercase first word, no trailing period. present tense
- one-line summary only. concise — name + signature already communicate, don't restate them
- `@expand` on classes/interfaces whose members ARE the API (e.g. State, Time). members listed alphabetically with param names in signatures
- `@example` for most callable exports. bare snippet, 2-4 lines, no imports/setup. skip only when the call is trivially obvious from name + params alone
- no JSDoc = not in reference. omit JSDoc from internal methods to hide them from the public API surface
- getters render as properties (no parens), methods show param names: `.query(terms)` not `.query()`
- return types auto-link to other exports in the same table when the type name matches

### When tabs don't fit

Not every page needs all three tabs. Skip tabs that would be empty or forced. If a topic is conceptual with no UI or code workflow, a single tab or no tabs is fine.

## Writing

- Concise. Every sentence must contribute information
- No editorial language ("by the end of this page," "let's," "just")
- No forward references — don't mention concepts before introducing them
- Progressive disclosure — simple first, complexity builds
- Code-first — show, then explain if needed
- Headings use Title Case. Short descriptions and fragments use lowercase
- Voice: instructional. Follow `voice.md` conventions

## Boundaries

- Docs answer "how do I use this?" Rules answer "what constraints apply when changing this?"
- Rules point to docs for context. No duplication between docs and rules
