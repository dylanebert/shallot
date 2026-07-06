---
paths:
    - "docs/**/*.md"
    - "docs/manifest.json"
    - "examples/zoo/**/*.ts"
    - "packages/shallot/src/**/index.ts"
    - "packages/shallot/scripts/literate.ts"
---

# Module Hardening

How a module reaches release-grade for V1, and the projection that gets it there: **a doc page is
generated, never authored.** The module's source yields the reference (the API / CORE JSDoc tables) and
the Internals-tab internals; its zoo specimen, written as a **literate program**, yields the page lead and the
Code-tab walkthrough; a one-entry nav manifest carries only what can't derive (title, icon, order). Every
word lives once, beside the code it explains, so there's no second copy to drift. Harden one module at a
time, in one focused pass — sweeping each axis across every module separately re-touches each one and
starves it of attention. `extras/orbit` is the worked exemplar (its specimen, source blocks, manifest
entry, test, and API are the reference shape). **Reference:** Godot's `--doctool` (its class reference is
generated from engine source, contributors writing only prose into the generated skeleton — we extend the
same to the *teaching* half) and Knuth's literate programming (narrative interleaved with the code it
documents, the source the document's source of truth). The convention details live in `exports.md` (the
tier list + barrel rules) and `docs.md` (JSDoc + tab conventions); this rule is the projection itself plus
the checklist over them.

## A module is hardened when

- **Public API is clean.** Barrel = developer surface, `/core` = extension surface, internal stays dark
  (`exports.md`). No primitive-renaming aliases, no import-collision `as`. **Before removing any export,
  grep every tier for consumers — `src/`, `editor/`, AND `examples/`** — and confirm each hit is
  load-bearing, not dead code: a tier-spanning grep finds the reference, but only reading the call site
  tells you it's real. Orbit's `OrbitSmooth` is the worked case both ways — auto-managed derived state
  the system adds itself (membership doubles as its "snapped to target yet?" init signal), it *looked*
  public because the editor referenced it, but that reference was a no-op against a fresh build State, so
  it's internal. An export is public only when something outside its module actually depends on it.
  **The API is the docs: a reference row is an API decision.** The reference and the inspector render
  whatever the barrel and the component schema hold, so reshaping the public surface is the sanctioned
  way to fix the docs — a field a scene author can't act on doesn't earn a row, it moves into internal
  derived state (orbit's `flyActive` runtime latch → `OrbitSmooth`, which also stopped a latch
  serializing into scenes); a default that renders as float garbage gets a principled value in code
  (orbit's `±89°` pitch clamps, not `±(π/2 − 0.01)` rad); a field kind the reflection can't name gets
  named (`entity`, so a ref reads as `@name`, never a number). An ugly or misleading generated page is
  first an API smell, only then a projection bug.
- **Reference is complete.** Every release-public export carries JSDoc (the module's `docs:check` bare
  lines are empty), conventions per `docs.md` (lowercase, present tense, one line, `@example` on most
  callables, definition site not barrel).
- **Component fields are annotated.** Every field of a public component carries a one-line `/** … */` doc
  comment above its declaration. The build parses them into `editor/src/lib/fielddocs.json`
  (`scripts/fields.ts` `componentDocs`) — the inspector hover + docs UI reference read it, IDE hover reads
  the comment direct. `extras/orbit` is the worked shape; `check-fielddocs` (in `bun check`) gates drift.
- **The component summary is author-facing; structured facts are traits, not prose.** A component's
  first-line JSDoc is its summary (the build takes only line 1), shown in the editor's add-component
  picker + inspector to a non-coding author — so it follows the UI knowledge floor (`docs.md`): plain and
  behavioral, no engine-internal vocabulary. `Orbit` is the shape ("orbit camera controls: drag to rotate
  around a target, scroll to zoom"), not a from-inside-the-code restatement of what it drives; mechanism
  detail lives in the page prose. Structured facts a beginner needs at a glance — `requires` / `excludes`
  / `provides` (a component that satisfies another's `requires`, e.g. `Body` provides `Transform`) /
  `singleton` ("one per scene") — are declared as **traits** and render as editor chips from live
  reflection (`ecs/core`), never written into the summary prose.
- **Page is generated, not authored.** The module's page assembles from its literate specimen
  (`#doc:intro` + `#doc:code`), its module source (`#doc:dev`), and a nav manifest entry, plus the
  reflection-generated reference tables — never a hand-written `docs/{engine,standard,extras,editor}/*.md`.
  Converting a module means **deleting** its hand-authored page and adding the manifest entry; the
  projection (next section) is the whole of it.
- **Zoo specimen exists and teaches.** A minimal, readable program using the module's public API,
  compile-gated by `bun check`. It carries the page's `#doc:intro` lead + `#doc:code` walkthrough as
  literate prose, and is the strongest drift gate: rename an export and the specimen stops typechecking.
- **Tests sit at the right tier.** Covered per `testing.md` (unit / CRUD / reload), or for a render /
  physics module by the consolidated gym atom that owns it (`render`, `pile`, …) — never a new
  per-module atom.

## The projection: how a page assembles

`scripts/literate.ts` (`assemblePage`) builds a page's intermediate markdown — the same marker-laden
markdown a hand-authored page was, so the existing `docs.ts` marker-expansion + render turns it into an
identical `docs/dist/` artifact. The page has no authored file; it exists only as this projection. Three
inputs:

1. **The nav manifest** (`docs/manifest.json`) — one `PageEntry` per page, **nav chrome only** plus the
   two pointers the projection reads. `slug` (the dist path), `title`, `description`, `icon`, optional
   `order`; `source` (the module dir under `src/` — the API/CORE tables, the source link, the `#doc:dev`
   blocks); `specimen` (the dir under `examples/zoo/` — the `#doc:intro` + `#doc:code` blocks). Nothing
   else belongs here — no prose, no structure; if it can derive, it derives.
2. **The literate specimen** under `examples/zoo/<specimen>/` — `#doc:intro` blocks (the page lead) and
   `#doc:code` blocks (the Code-tab walkthrough).
3. **The module source** under `src/<source>/` — `#doc:dev` blocks (the Internals-tab internals).

The assembled page is the frontmatter, the `#doc:intro` lead, then the `#doc:code` walkthrough and the
generated `<!-- API:source -->` reference. Block order within a file is page order, and files are walked
sorted, so **file order is section order**: name and order the specimen's files the way the page should read.

**The tab shape follows the audience — three cases, set by what the page carries.** A page is single-tab
(no tab chrome) when it serves one audience and two-tab when it serves both:

- **author-only** (an `API` source, no `#doc:dev`) — walkthrough + the `<!-- API: -->` table inline. The
  leaf shape (`orbit`).
- **extender-only** (a `CORE` source with `#doc:dev`, but no author `API` source) — walkthrough + the
  `#doc:dev` internals + the `<!-- CORE: -->` table inline. Its whole audience is the extender, so a
  Code/Internals split would leave one tab empty (`standard/surfaces`).
- **dual-audience** (an author `API` source *and* `#doc:dev`) — a two-tab group, a **Code** tab
  (walkthrough + API) and an **Internals** tab (the `#doc:dev` blocks + CORE). The common module page
  (`audio`, `gltf`, `standard/rendering`).

Write `#doc:dev` only when the module has a real **extension surface**, a `/core` API an extender builds
against, and document the use case (ideally shown in a zoo specimen) of extending it, not how the system
works inside. A leaf module like `orbit` (no `/core`, not extended) is author-only; a maintainer-facing
"why it's built this way" is a plain code comment, not a docs page. Don't write `#doc:dev` mechanics just
to fill a tab.

**Which tab a reference marker lands in is fixed by which carries it:** `<!-- API: -->` (the public *what*,
under the walkthrough's *how*) in the Code surface, `<!-- CORE: -->` (the `/core` extension surface) in the
Internals tab. The projection places both; you write neither.

**There is no per-module Editor tab, by decision, not deferral.** The no-code author's per-module surface
is generated from the same source the page is: the component's line-1 JSDoc summary in the add-component
picker, the field docs in the inspector hover (`fielddocs.json`), and `docFor(component)` deep-linking the
inspector to the page — plus the page itself opening scene-first (the lead section is the scene snippet,
the complete no-code path, before any TypeScript). Per-module editor-workflow prose ("add the component,
tweak fields") would read the same on every page, which is `docs.md`'s substrate test: it belongs to one
guide page, not N tabs. Don't hand-author an Editor tab and don't add an editor block kind to the grammar.
The full audience↔tier mapping lives in `docs.md` "Tabs".

## The `#doc:` convention

A literate block is a `// #doc:<kind>` marker line followed by its prose as contiguous `//` comment lines
(the `//` stripped, a bare `//` → a blank line), ending at the first non-comment line or the next marker.
A markdown heading inside the prose (`// ### Fly Mode`) is prose, not a terminator — sub-section the
walkthrough with headings freely.

- **Kinds.** `#doc:intro` and `#doc:code` live in the **specimen**; `#doc:dev` lives in the **module
  source**. (`assemblePage` reads intro/code only from the specimen dir, dev only from the source dir, so
  a misplaced kind silently drops — keep each in its home.)
- **Paired snippet.** A block optionally shows code beside its prose, emitted as an `<!-- EXAMPLE:path -->`
  marker the build expands. Pair it two ways: an explicit `#doc:code source:<path>` (zoo-relative, e.g.
  `source:orbit/public/scenes/orbit.scene` — for the scene, the manifest, a cross-file snippet), or a
  trailing same-file `// #region <name>` directly after the block (the block pairs with that region of its
  own file). A block with neither is prose-only.
- **Routing is the specimen's home module.** Today one specimen feeds one page (its manifest `specimen`),
  so a `#doc:code` block routes to that page implicitly. The many-to-many tag (one specimen's regions
  feeding several pages) is deferred — see *Routing* below.

The specimen file reads top to bottom as the page does: `#doc:intro` first, then `#doc:code` blocks
interleaved with the regions they explain. `examples/zoo/orbit/src/tune.ts` is the worked shape.

## Write the walkthrough code-first

The page is the code; the prose points at it. Svelte's `$state` page is the bar (`docs.md`): open with
code, explain only what isn't obvious, reference at the bottom. The first orbit draft inverted this, 1300
words narrating mechanics against three snippets, and it's the failure mode to copy away from. The shape
that replaced it follows `docs.md` for voice and for what earns a place (concise, neutral, em-dashes off;
shared substrate is a stub plus a link, not a re-teach). The projection-specific points:

- **A `#doc:code` block is one snippet plus one to three sentences.** Lead with the snippet (a `source:`
  file or a `#region`), then say only what the code doesn't show: the result, the gotcha, the one knob that
  matters. **A walkthrough section exists only anchored to a snippet** — a heading over a lone sentence is
  cut, three ways: fold the behavior into the snippet itself (orbit's scene grew `target: @box`, killing a
  follow-a-target stub), ride it as a one-sentence prose-only block directly after the snippet it qualifies
  (orbit's fly-controls line under the tune plugin), or leave it to the JSDoc the reference already renders
  (orbit's speed-readout stub restated `OrbitOverlayPlugin`'s summary — a Two Truths leak). Prose with no
  snippet running past a sentence is the smell.
- **Mechanics belong in `#doc:dev`, never `#doc:code`.** The walkthrough teaches the usage path. How it
  works (the state machine, the reproject-on-exit, the precedence rule) is Internals-tab material, and it
  belongs in `#doc:dev` as a named decision pointing at the code, not a line-by-line retelling. Orbit
  bloated because the mechanics lived in both Code and Internals, in full.
- **Document the feature, not the example.** The specimen is scaffolding that makes the snippets compile
  and gates drift; it is not the subject. Show how a user would use the module in any project, not how this
  example is wired. Orbit's `OrbitTune` plugin is a real pattern worth showing as code; the example's own
  `shallot.json` (with its teaching plugins) is plumbing, so the page links plugin-enablement to its guide
  instead of pasting the manifest.
- **Don't enumerate fields in prose.** The reference table (and the deferred FIELDS table) carries the
  per-field schema from JSDoc; the walkthrough teaches the common path and names the few fields it sets.
  Restating every field in prose is the verbosity trap and a `docs:check` leak (Two Truths, `docs.md`).
- **Whole-file snippets must read clean.** A `.scene` / `.json` `source:` has no region slicing; it
  extracts entire, so the specimen's scene and manifest are themselves the snippet. Keep them minimal: no
  editor round-trip noise (a `transform` of `rot: -1.13e-8 …` is float garbage, not a teaching value), no
  entity the page doesn't reference.

## Zoo specimens

`examples/zoo/` is the docs' teaching tier (one of the three example tiers, beside gym / showcase): one
minimal **editor-openable project per module** — `examples/zoo/<module>/shallot.json` (the project
manifest: a `scene` under `public/scenes/` + `plugins` enablement) plus any plugin modules under `src/`,
the canonical project shape (`shallot <dir>` opens it in the editor, `shallot dev <dir>` runs it
standalone; no `index.html`/`main.ts`/`vite` needed — the CLI supplies that scaffolding). A specimen is
**usage truth** — it keeps the docs honest about the code, the way a gym atom keeps the engine honest
about itself, and now also carries the page's teaching prose. The root `tsc` (in `bun check`) is the
compile gate — no harness verdict, no `assert`.

A specimen earns its drift gate by **using the module's API in code, not only in scene attributes**: a
scene is untyped, so a declarative-only specimen wouldn't break when an export is renamed. Reference the
component in a small plugin module the manifest declares (orbit's `src/tune.ts` plugin sets
`Orbit.sensitivity` / `Orbit.flySpeed` in `warm`) so a rename fails `tsc`. That module is also where the
`#doc:code` walkthrough lives, so the prose sits beside the API it teaches. Keep tuning that fights the
editor out of code — pose fields belong in the scene (the editor edits them there); the tune plugin
carries feel knobs the editor doesn't author, and a `#doc:code source:<scene>` block shows the scene.

Keep a specimen tight: the one concept, real defaults, no unrelated scaffolding. It is read by a beginner
and projected into a page, so it earns the same readability bar as `standard/`. `examples/zoo/orbit/` is
the worked exemplar.

## The reference is kind-aware and reflection-generated

The projection emits the markers; `bun run build` fills them from source, so nothing in a table is
hand-written and nothing can drift. `<!-- API:source -->` (Code tab) renders the barrel's public exports,
`<!-- CORE:source -->` (Internals tab) the `/core` extension surface. Each entry carries a **kind badge**
so a reader sees what an export *is* at a glance, and the kinds that hold structure expand to show it
instead of a bare name and summary. What each kind renders, so the reference is useful per type:

- **component** → its field table (field, type, default, doc). Type and default are reflected from
  `schema()` (the editor inspector's rows); the meaning is the field's JSDoc, which is why every field
  earns a doc comment.
- **enum** → its option → value table, parsed from the `as const` literal, so the reader sees the legal values.
- **plugin** → the components, systems, and dependencies it bundles, each component also documented on the
  page linked to its entry, so the reader sees what enabling it brings in.
- **function** → its signature (params) and JSDoc, `@example` included.
- **type / class** → its summary; `@expand` lists a class's documented members.

A dead marker (its module gone) logs a `docs:check`-gated `warning:`. `orbit` is the worked shape: `Orbit`
reads as a component with its field table, `OrbitMode` an enum with Free/Locked, `OrbitPlugin` a plugin
listing the linked `Orbit` component, `OrbitSystem`, and its dependencies.

The standalone `<!-- FIELDS:component -->` marker still renders a component's field table on its own (for
an explicit placement on a hand-authored page, e.g. a guide); a page that lists a component in its API reference gets
the fields there already, so it needs no FIELDS marker.

## Routing: one-to-one and many-to-many

A `PageEntry` names `source` (author API modules), optional `core` (extension modules — defaults to
`source`), and `specimen` (zoo dirs). Each is a string (the one-to-one degenerate — orbit's) or a list.
Three routing levers cover every page shape:

- **`page:<slug>` on a block** (`#doc:intro` / `#doc:code` / `#doc:dev`) routes it to one page when its
  specimen or source feeds several. An untagged block routes to every page listing that specimen/source
  (the one-to-one case). This is how one composition specimen (`kitchen`) splits its blocks across the
  pages it teaches — the rendering blocks route to `standard/rendering`, the surface-authoring blocks to
  `standard/surfaces`.
- **`source` as a list** draws a conceptual author page from several modules (`standard/rendering`:
  `render` + `part` + `sear` + `glaze` — one Reference with a source-tagged API table per module).
- **`core` decoupled from `source`** picks which module's `/core` renders the CORE table + `#doc:dev`
  internals, independent of the author API list. So the rendering page lists four author modules but draws
  its extension story from one (`core: standard/render`, not `sear`/`part`/`glaze`), and an extender-only
  page (`standard/surfaces`) draws a CORE table from a module (`core: standard/sear`) it lists no author
  API for. When `core` is omitted it defaults to `source` (a module documents its own `/core`).

`kitchen` is the worked case for all three: one specimen + a multi-module `source` + a decoupled `core`,
its blocks split by `page:` across the rendering + surfaces pages. A module surfacing a *new* routing need
refines these levers, never grows a parallel mechanism. The standard is living.

## Comment-narration lint exempts `#doc:` blocks

The `Checking comments…` PostToolUse lint (`kex/.claude/hooks/check-comments.ts`) flags comments that
narrate an edit. `#doc:` prose is page text carried in comments, so a line like "Now a mouse-drag
rotates…" is correct there but trips the lint's `^now` tell. The hook exempts every line of a `#doc:`
block (the marker plus its contiguous comment prose, mirroring `parseDocBlocks`). When writing a
specimen's literate prose, page voice (`docs.md`) governs, not the comment rule.

## Working the sweep

`bun run build`, `bun run docs:check --update`, `bun check`, the zoo compile gate are the per-module
validation. `docs:check` runs the build fresh and gates the code↔prose contract: objective failures (a
marker whose module is gone, a dead API/CORE table, a `source:` that doesn't resolve) always fail; the
ratchet classes (`bare` — a JSDoc-less public export, `hole` — a tier subsystem with no page, `keep` — a
prose mention of a generated reference name) diff against `docs-baseline.txt`, which the docs pillar burns
to empty. The standard is living: a module that surfaces a case the checklist doesn't cover refines this
rule, not a special case for that module.
