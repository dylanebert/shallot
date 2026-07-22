# red-box — expected artifacts (withheld)

The problem shape from `examples/recipes/build-a-scene` and `stylize-the-look` (color): a static scene
with a verifiable visual claim.

What a correct project does:
- one `Part` at the origin, `color="rgba: <red>"` (r high, g/b low), a camera facing it, a dark
  background/ambient so the cube reads against it
- no `simulation`-group system rotating anything — the scaffold ships a `Spin` system; a correct
  answer removes it (or its `shallot.json` entry)

Gate observation: centre 20% region is red and brighter than the corner; two frames 1.2s apart barely
differ (static). Purely pixel-based — the authoring path doesn't matter.

Common failure modes this catches: left the spinning cube (fails "static"); changed nothing (orange,
fails "red"); made the whole background red (fails "stands out").
