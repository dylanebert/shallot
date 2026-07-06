import { image, type Plugin } from "@dylanebert/shallot";
import { meter } from "./meter";

// #doc:intro
// Textured quads that live in the world and turn to face the camera: icons, markers, and labels
// pinned to your entities.

// #doc:code source:sprite/public/scenes/sprite.scene
// Give an entity a `sprite` and it draws a billboarded quad at the entity's transform. `image` names a
// registered image, `size` is world units, and `anchor` is the pivot: `0.5 0` pins the bottom of the
// marker to the post beneath it. `billboard: screen` keeps it facing the camera as the view orbits.

// #doc:code
// Register each image once, by name, before the scene loads. A plugin's `initialize` is the place; the
// name you pass is what the scene's `image:` resolves. `image` takes a url or a `Blob`.
// #region register
export const Icons = {
    name: "Icons",
    initialize() {
        image("/icons/pin.png", "pin");
        image("/icons/ring.png", "ring");
    },
    systems: [meter],
} satisfies Plugin;
// #endregion

export default Icons;
