---
paths:
    - "docs/**/*"
---

# Docs

Doc pages optionally map to a code folder via `source` frontmatter (shows source link + path). It must resolve to a real folder under `src/` — it names the module the page documents (`standard/render`, `extras/orbit`) and rots silently when that folder moves or renames, so the `docs:check` drift gate validates every `source` resolves. Omit `source` for pages that don't correspond to a specific module (e.g. quick-start). Code is the source of truth. Make the code elegant first, then document what's non-obvious.

`docs/` is for game developers using Shallot. `.claude/rules/` is for engine developers modifying Shallot.

## Structure

- **Start** (`docs/guide/`) — progressive disclosure, builds concepts step by step
- **Engine** (`docs/engine/`) — ECS, scenes, app, utils
- **Standard** (`docs/standard/`) — built-in subsystems
- **Editor** (`docs/editor/`) — editor-enabling engine libraries (document model, session sync) and Svelte editor integration
- **Extras** (`docs/extras/`) — optional plugins

Generated reference tables come from two markers, both replaced by `bun run build`: `<!-- API:path -->` (the public API table, in the **Code** tab) and `<!-- CORE:subsystem -->` (the core/extension API table, in the **Internals** tab). All three tabs are otherwise authored prose, optionally embedding extraction markers (`<!-- EXAMPLE: -->`, `<!-- FIELDS: -->` — see `hardening.md`) that inject generated content.

`bun run build` renders each page to HTML (marked + Shiki) and writes it to `docs/dist/`, frontmatter kept verbatim and tab markers (`<!-- tabs -->`, `<!-- pick -->`) preserved as comments. So `docs/dist/` is render-ready, not markdown — consumers (the site, the editor docs panel) split the markers into their tab UI and display the HTML directly. Author in markdown under `docs/`; never edit `docs/dist/`.

## Tabs

Three audience tabs, three readers: **Editor** (the author, editor workflows), **Code** (the developer, the TypeScript API), **Internals** (the extender, internals + the core API). The three readers map onto the export tiers (`exports.md`): the editor author's *per-module* surface is generated in-editor from the same JSDoc (component summary in the add-component picker, field docs in the inspector hover, `docFor` deep-links) plus the page's scene-first lead — a projected module page emits no Editor tab (`hardening.md`); the Code tab is the barrel; the Internals tab is the module's `/core`. Tab groups are atomic blocks wrapped in `<!-- tabs -->` / `<!-- /tabs -->`. Inside, use `<!-- tab: Name -->` for each. Tab groups can appear anywhere in the page, multiple times. Content outside tab groups is shared across all tabs.

Inline tabs use `<!-- pick -->` / `<!-- /pick -->` with `<!-- pick: Name -->` for each option. Used for scene/code alternatives and OS-specific instructions. OS tabs (`<!-- os -->`) also work — both render the same way. If tab names match an OS ("Windows", "Mac/Linux"), the visitor's OS is auto-detected.

Title and description render from frontmatter above all content, not from the markdown h1.

### Editor

Audience: someone who doesn't write code. Should not feel redundant for someone who does.

- Knowledge floor: assume no code and no engine vocabulary. A concept that only exists in code (a component, a system) gets named in interface terms or left to the Code tab — never assume a TS reader
- Editor workflows only. No TypeScript
- Accessible but not condescending. Don't simplify concepts, simplify the interface to them
- A reader should be able to follow only Editor tabs and understand simple usage
- Can point to the Code tab for more customization
- For topics where the editor isn't relevant (e.g. Compute), keep a short conceptual explanation and point to Code. Don't force Editor content where it doesn't fit

### Code

Audience: game developers. The primary tab for most topics.

- Knowledge floor: assumes TypeScript fluency and general game-dev vocabulary (mesh, camera, transform, component) — don't explain those. Explain Shallot-specific concepts (the ECS shape, plugins, the scene format) on first use, then link the engine page rather than re-teaching ECS
- TypeScript API, scene files, systems, plugins
- Can cover the same flows as Editor but with scene files and code
- Extends beyond what the editor can do — custom systems, novel gameplay behavior
- The public API table (`<!-- API:path -->`) renders at the bottom, the generated *what* under the hand-written *how to use*. Entries are kind-tagged and the structured kinds expand (a component to its field reference, an enum to its options, a plugin to its parts), so prose teaches usage and the table carries the schema. Per-kind detail: `hardening.md`

### Internals

Audience: the extender — someone writing a custom pipeline, producer, or tool against a module's `/core` surface.

- Knowledge floor: assumes the engine architecture (the ECS/GPU model, the render contract). Explain only this module's internals; link the contract page (`render.md`, `ecs.md`) rather than restating it
- Internal-workings context: architecture, buffer layouts, registry internals, ordering anchors
- The core API table (`<!-- CORE:subsystem -->`) renders at the bottom — a module's `/core` extension surface (the `*/core` tiers in `exports.md`)

**Reference tables are auto-generated from JSDoc** (the `API` and `CORE` markers). Entries render as a scannable list; `@expand` classes/interfaces show members inline. JSDoc conventions (on definition site, not barrel):

- lowercase first word, no trailing period. present tense
- one-line summary that says what the export *does* — the semantics, edge behavior, sentinel/zero meanings (`0` = world origin), valid range, when it has no effect. Never a wordier restatement of the name; the name + signature already carry that. The JSDoc is the IDE-tooltip surface where a developer meets the API, so a constraint that lives only in a page is invisible at the point of discovery. `Orbit`'s field docs (`extras/orbit`) are the worked shape — "follow damping, 0–1; higher snaps to the target pose faster", not "the smoothness value"
- `@expand` on classes/interfaces whose members ARE the API (e.g. State, Time). members listed alphabetically with param names in signatures
- `@example` for most callable exports. bare snippet, 2-4 lines, no imports/setup. skip only when the call is trivially obvious from name + params alone
- no JSDoc = not in reference. omit JSDoc from internal methods to hide them from the public API surface
- getters render as properties (no parens), methods show param names: `.query(terms)` not `.query()`
- return types auto-link to other exports in the same table when the type name matches
- `{@link Name}` in a summary renders as a link to `Name`'s same-page entry (plain code when the page has none); the editor field-hover strips it to the bare word. It resolves **only in an entry's own JSDoc summary** — a component field-doc row and `#doc:` prose render it raw, so name the reference with a plain backtick there (orbit's field docs are the shape)

### When tabs don't fit

Not every page needs all three tabs. Skip tabs that would be empty or forced. If a topic is conceptual with no editor or code workflow, a single tab or no tabs is fine.

## Two Truths

The reference is two things, each with exactly one home: JSDoc in code, prose in markdown. They describe the same exports from different angles and must not overlap.

- **Educational prose (the Editor / Code / Internals tab bodies) says *why* and *how to use*.** The convention that makes a field linear, when to pick `aim` over `lookAt`, the workflow that calls `parse` then `load`. It names exports freely — you can't teach usage without naming the thing.
- **The generated reference tables carry *what*.** Each export's signature and one-line behavior, generated from its JSDoc — the `API` table in Code, the `CORE` table in Internals. Never hand-written.
- **Neither restates the other.** Naming an export in prose is fine; restating its summary or signature is the leak. "Call `whenLoaded(id)` before reading the buffer" is how-to-use. "`whenLoaded(id)` awaits a sample's pending decode" copies the JSDoc — cut it. The same leak hides in a hand-maintained signature or field table; the Keep-or-cut test below has the fix.

When the generated entry is *bare* (the export has no JSDoc), prose describing its behavior is a JSDoc gap, not a duplication. The fix is to write the JSDoc so the marker carries it, then delete the prose — not to leave the *what* stranded in hand prose because the Reference is empty.

**Keep-or-cut test.** Finding the candidates is mechanical — the `docs:check` leak detector surfaces every prose mention of a generated `ref-*` name. It can't judge intent; you do. One test forces the call: **blank the export's name out of the sentence — does anything survive?**

- keep — "call `whenLoaded(id)` before reading the buffer" → "call ___ before reading the buffer" still teaches the ordering. The content is the *workflow*; the name is just the referent.
- cut — "`whenLoaded(id)` awaits a sample's pending decode" → "___ awaits a sample's pending decode" is the JSDoc summary and nothing else. The Reference already generates it.
- cut — a hand-maintained field or signature table (defaults, descriptions, `(args)`) restates the Reference in another shape. `@expand` the type so its JSDoc renders the members, then delete the table.

The page reads clean when prose and Reference don't overlap on *what*.

## One page per concept

A page documents what is specific to its module. Everything shared lives in one place and is linked, not
re-taught. Three kinds of information, three treatments:

- **Specific to this module.** Its API, behaviors, configuration. The page's job; write it in full.
- **Shared substrate.** Standard across modules: enabling a plugin, the scene and manifest format, the ECS
  model, adding a system, the platform floor. One concept page owns it; every other page assumes it or
  names it in a clause and links the owner, never re-teaching it. Re-teaching is N copies to maintain and N
  places to drift.
- **Related, owned elsewhere.** A concept a neighbouring page documents. Link it, don't recap it.

The test: would this sentence read the same on every other module's page? Then it is substrate. Cut it to a
stub and link the page that owns it. A page that re-explains how to enable a plugin, or what a system is,
has stopped documenting its module.

**Cross-page links** use `[text](doc:slug#anchor)`: `slug` is the target page's dist path
(`guide/quick-start`), `anchor` optional. The build leaves the `doc:` href in place; the site and the
editor reader each resolve it to their own routing (a hard `/`-rooted URL can't, since the two route docs
differently). Use it for prose links and cross-page references alike.

## Coverage

The release-public surface is the tier list in `exports.md` — the `@dylanebert/shallot` barrel, `/extras`, `/editor`, `/runtime`, and every `/*/core` subpath (the `package.json` `exports` map is its machine-readable form). That is the one source of truth for what ships; this is the documentation it obligates, not a second copy.

- **Every release-public export carries JSDoc.** An export reachable through a tier with no JSDoc generates a *bare* Reference entry — name and signature, no description. Bare is a gap to fill, not a resting state.
- **Every subsystem in a tier has a page.** Each barrel and `/*/core` subpath maps to one page under the matching `docs/` folder (`engine/`, `standard/`, `extras/`, `editor/`). A public subsystem with no page is a coverage hole.
- **Internal stays dark on purpose.** Code reachable only within a module carries no JSDoc and no page — that omission is how it stays off the public surface (`exports.md` "Barrel rules"). Coverage binds the shipped tiers only; don't document a test seam or an internal helper into visibility.

The `docs:check` drift gate mechanizes all three (bare-entry, subsystem-has-page, `source` resolves). The obligations hold whether or not the gate has flagged a given case.

## Writing

Voice: instructional, second person, present tense. Confident and plain — a competent peer explaining the thing, not a brochure and not a man page. The `voice.md` AI-tells and banned-word list still apply, and so does its guard against flat affect: clarity, not recitation.

- **Open with what it does.** A page's and a section's first line states what the thing does and the problem it solves, before any mechanism. Lead with the function, not the setup.
- **Anchor the unknown to the known.** Introduce a concept by its closest familiar analogue, then state the difference — it lets you skip re-explaining the familiar half (`voice.md` "Progressive disclosure" owns the technique + guard).
- **Code-first, and show the result.** Show the snippet, then explain only what isn't obvious from it. A snippet isn't done until the reader can see what it produces — an inline `// =>` comment on the output, or one sentence on the resulting behavior. Order snippets by what's taught, not by the API.
- **Specific guarantees, not vague claims.** Replace a quality adjective with the property it stands for. State limits plainly: what the feature does *not* do, what's partial, the failure mode. Honesty about limits is what keeps "confident" from sliding into a brochure.
- **Verify claims.** A behavioral claim in prose traces to the specimen/test, or you ran it. Code snippets are verified by construction (extracted from a compiling specimen); prose claims aren't — check them. Preview the rendered page (`bun run build`): a broken marker or unbalanced fence silently swallows everything after it.
- **Concise, in order.** Every sentence contributes information. No editorial throat-clearing ("by the end of this page," "let's," "just"). No forward references — don't name a concept before introducing it. Progressive disclosure — simple first, complexity builds.
- **Headings:** Title Case for page/section headings, lowercase for short descriptions and fragments. In `docs/guide/` (the on-ramp spine), section headings are action-object ("Add a camera", "Move an entity"); per-module reference pages keep noun headings.

## Boundaries

- Docs answer "how do I use this?" Rules answer "what constraints apply when changing this?"
- Rules point to docs for context. No duplication between docs and rules
