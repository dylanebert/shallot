import { resolve } from "node:path";
import { componentDocs, serializeDocs } from "./fields";

// editor/src/lib/fielddocs.json is a committed build artifact: the source-parsed UI-reference data the
// editor inspector imports (the browser can't read source). Regenerated here by `bun run build`; the
// `check-fielddocs` gate (in `bun check`) fails if the committed copy drifts from the live source.
const out = resolve(import.meta.dir, "../editor/src/lib/fielddocs.json");
const docs = await componentDocs();
await Bun.write(out, serializeDocs(docs));
console.log(
    `  → field docs for ${Object.keys(docs).length} components to editor/src/lib/fielddocs.json`,
);
