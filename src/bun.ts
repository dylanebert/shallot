import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

plugin(
    typegpu({
        // Keep dependency JavaScript out (the hook breaks picomatch's default export), while reaching
        // Shallot's source in node_modules as well as the project's own TypeScript and JavaScript.
        include:
            /^(?:.*\.(?:[cm]?ts|tsx)|(?:(?!.*[/\\]node_modules[/\\]).*|.*[/\\]node_modules[/\\]@dylanebert[/\\]shallot[/\\]src[/\\].*)\.(?:[cm]?js|jsx))$/,
    }),
);
