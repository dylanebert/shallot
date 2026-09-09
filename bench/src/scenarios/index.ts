// Importing a scenario module runs its `register(...)` — adding a scenario is a new file here plus one
// line in this barrel, which `main.ts` imports for the side effects. What each scenario covers, and why,
// is the per-scenario `covers` table in `timeouts.ts`; what each one is for is that scenario's own file.
import "./accel";
import "./backend";
import "./cells";
import "./chain";
import "./character";
import "./constraints";
import "./gltf";
import "./gpu-diagnostic";
import "./mesh-fixture";
import "./motor";
import "./outline";
import "./pile";
import "./queries";
import "./raining";
import "./render";
import "./rotation";
import "./sat";
import "./sprite";
import "./stress";
import "./text";
