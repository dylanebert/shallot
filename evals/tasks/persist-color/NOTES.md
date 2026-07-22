# persist-color — expected artifacts (withheld)

Problem shape from `examples/recipes/save-and-restore` + `respond-to-input`: an input-driven state
change that survives a reload through the serialize/restore (or `localStorage`) path.

What a correct project does:
- a single `Part` cube; a system reading edge-triggered `Digit1`/`Digit2`/`Digit3` presses that set the
  entity's `Color` to red/green/blue
- the chosen colour is stashed (its scene serialized to `localStorage`, or the colour value itself) and
  read back on boot, so a fresh load restores it without any input

Gate observation: after a synthetic click + number-key press the centre reads the pressed hue; after a
fresh navigation (same origin, `localStorage` intact) with no further input the centre still reads that
hue. The target hue is chosen to differ from the scene's starting colour, so a project that resets to a
fixed default on reload fails the persistence assertion rather than passing by luck.

Common failure modes this catches: keys do nothing (fails "pressing the key paints it"); colour set in
memory but never persisted (paints correctly, then reverts to the default on reload — fails
"still … after a reload"); scene hard-coded to one colour from the start (the target-differs-from-start
choice makes the paint assertion fail).
