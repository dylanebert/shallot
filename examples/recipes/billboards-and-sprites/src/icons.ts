import { image, type Plugin } from "@dylanebert/shallot";
import { meter } from "./meter";

// a sprite draws a billboarded quad at its entity's transform: `image` names a registered image,
// `size` is world units, `anchor` is the pivot, and `billboard: screen` keeps it facing the camera.
// register each image by name before the scene loads — a plugin's `initialize` runs pre-parse, and
// `image` takes a url or a `Blob`
export const Icons = {
    name: "Icons",
    initialize() {
        image("/icons/pin.png", "pin");
        image("/icons/ring.png", "ring");
    },
    systems: [meter],
} satisfies Plugin;

export default Icons;
