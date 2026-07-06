import type { Plugin, State, System } from "@dylanebert/shallot";

// A minimal external plugin library — the install-test's proxy for a real third-party plugin a user
// installs and references from a manifest by subpath (`"Widget": "shallot-widget-fixture/widget"`). The
// subpath's default export is the Plugin (the package declares its entry, per the manifest contract).
const WidgetSystem: System = {
    group: "simulation",
    update(_state: State) {},
};

const WidgetPlugin: Plugin = { name: "Widget", systems: [WidgetSystem] };
export default WidgetPlugin;
