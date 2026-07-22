# color-on-key — expected artifacts (withheld)

Problem shape from `examples/recipes/respond-to-input` + `game-loop`: an edge-triggered input driving
a state change.

What a correct project does:
- a white `Part` at the origin; a system reading an edge-triggered spacebar press that sets the
  entity's `Color` to green (and leaves it green)

Gate observation: the centre reads white before any key; after a synthetic `Space` press it reads
green (g dominant). The change is bound to the input — a scene that is green from the start fails the
"starts white" assertion.

Common failure modes this catches: nothing wired to the key (stays white); color toggles per-frame
instead of latching (may be caught by the flat 500ms settle); cube green from the start (fails "starts
white").
