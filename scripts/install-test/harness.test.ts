import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { CANONICAL_OUTPUT_FILES, canonicalInputFiles } from "./harness";

const EXPECTED_INPUTS = [
    "../shallot/scripts/projections.ts",
    "assets/icon-1024.png",
    "bin/build.ts",
    "bin/bun-native.ts",
    "bin/cli.ts",
    "bin/dev.ts",
    "bin/features.ts",
    "bin/gpu-globals.ts",
    "bin/native.ts",
    "bin/recipe.ts",
    "bin/run.ts",
    "bin/scaffold.ts",
    "bin/toolchain.ts",
    "bin/tui.ts",
    "bin/tui/color-support.ts",
    "bin/tui/cursor.ts",
    "bin/tui/diff.ts",
    "bin/tui/encoder.ts",
    "bin/tui/index.ts",
    "bin/tui/resize.ts",
    "bin/tui/screen.ts",
    "bin/tui/sgr.ts",
    "bin/tui/terminal-model.ts",
    "bin/tui/types.ts",
    "bin/verify.ts",
    "package.json",
    "rust/window/.cargo/config.toml",
    "rust/window/Cargo.lock",
    "rust/window/Cargo.toml",
    "rust/window/build.rs",
    "rust/window/icon.ico",
    "rust/window/icon.rc",
    "rust/window/src/cef_backend.rs",
    "rust/window/src/helper.rs",
    "rust/window/src/mac.rs",
    "rust/window/src/main.rs",
    "rust/window/src/wry_backend.rs",
    "scripts/build.ts",
    "src/harness/browser.ts",
    "src/project/assets.ts",
    "src/project/command.ts",
    "src/project/engine.ts",
    "src/project/generate.ts",
    "src/project/host.ts",
    "src/project/manifest.ts",
    "src/project/vite.ts",
].sort();

const EXPECTED_OUTPUT_HASHES: Record<string, string> = {
    "assets/icon-1024.png": "be7b7331eed3c1c3de83c381282ffa699e3cef6269aa3d82649832394dcbf311",
    "bin/build.ts": "c6078db74169c85f0ebc1fc0fc02669e7cc5cabcb9955d8e3bcdc4fe39fd17c2",
    "bin/bun-native.ts": "eb2cd49ff52d7d597d83fb0eabdbc324c76410b09d450657f8317567c0e5f233",
    "bin/cli.ts": "b31722eac75f93d90bb506ede1bacf9e55500eb97d3e8e71ea17c513d9af30c2",
    "bin/dev.ts": "943f14c7cb280cd7ad907ce6c3cc3ee6e8ad7b7fcf4ecc8f7dd4097082553b94",
    "bin/features.ts": "cbb2e4429908f2dd82875a5783b000d6452faf1f2f9d9371a22238a42c4e4365",
    "bin/gpu-globals.ts": "59c6ab0cd3593d77ab337e09a7890cbb67a5c3a08193e6ebfe2e889745f5d982",
    "bin/native.ts": "229a625751f31d3900aac314196945e91512d40cebe934d7288ea258844fc18c",
    "bin/recipe.ts": "5a783463cdec968b125d995efc904d85434cd941505699efca0c26f4a6ff938b",
    "bin/run.ts": "818ea4233bb5151f14bf288440e9f1cd4307483c266a67a313b6baad6e88dfbd",
    "bin/scaffold.ts": "b92bf9f67ced3574021ae55e83abe17ef6ead4d9b62f5999456cf8ab6c052711",
    "bin/toolchain.ts": "2620ce4d710b032ddc2890a0c6b424f9bf89ef8ed48ab5ebddf3eb01a67b9b3e",
    "bin/tui.ts": "4c5c6c1c7d783b17e295d315bc8f17cbe571df6370eff6130f8185ae6e3af28c",
    "bin/tui/color-support.ts": "85232126e3e4f5a3372f0877831a4c98c4c46e8417b7318b5ba9c64b5a8babb8",
    "bin/tui/cursor.ts": "6efc366d65f438c89907deaecc3b5e17f07ea57d866f56067e731fbc0363b83b",
    "bin/tui/diff.ts": "fa6f00502c6c3b01dbb921aaa7e0a16982cc67245087584e49900b50d67b37b0",
    "bin/tui/encoder.ts": "5a8dabdb2fbd6560ef131b92e37193b85ea17b2ed989640ef13d72d59d35a11b",
    "bin/tui/index.ts": "e1e06425d76dc270b66762c0f158dd9a529b92d214c495aa30e7060e6c093497",
    "bin/tui/resize.ts": "c44951b46ebc616e2a1852567f68ebef218f9921ce7734aecd37e115676ee8b8",
    "bin/tui/screen.ts": "073b94f272fd7697a73d7e117e3374bc1bd71afbb19d79602d9672d42abf2ceb",
    "bin/tui/sgr.ts": "1f5a64e02cff502c6062a8f1991670be9358ca1df3fdcdd87ecbfc52e6b33df2",
    "bin/tui/terminal-model.ts": "1832ce90c034e15363ed4001e73ddbf263871092ef1b51e9433231e5da423ba6",
    "bin/tui/types.ts": "84761c3e5047356708681c6a0db155affe24fa375deb855e71fe6455136335b1",
    "bin/verify.ts": "aa4888ea45a5e8c90a05afbeb20d4c1bca0be1ba09c3f95c1678e0203993b6e3",
    "dist/bun-webgpu-LICENSE": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    "dist/bun-webgpu-NOTICE": "038daa236eb4671927f5d5cc40f63156351d746e1ce0f86ce4f0771c2c900f41",
    "dist/harness-browser.js": "bf07c6f2b3d15da4943122d8ce5c52d3a218573386abd7f1cee5b24fe414a070",
    "dist/native.js": "2f573c74b9f8cb96aa86c17e7a4c47ab9b210edd8335526126e40c8d8288d86c",
    "dist/vite.js": "e2a66136eac4d6363a22b9cfead2b97d3b8ab067472056262485069a0e793207",
    "rust/window/.cargo/config.toml":
        "16b8bc422a0332f72af131db0a1347bb4f8cdeaf445578e051035ca4cbfd0f7c",
    "rust/window/Cargo.lock": "cf0aee4676d1f2fcd5a6d13da3a7853af19b35eef897e544348dca52021b85eb",
    "rust/window/Cargo.toml": "3b5cc0bbb3b0655d9899f68ef894ed06e51926deb64e66ab4c3a583dceb3eb14",
    "rust/window/build.rs": "9f88b5c8325edef7b52ae620333940f2450f171d4de26f0b660988827d4c2c84",
    "rust/window/icon.ico": "b200af96feeefefe455d5599c9216a3d39e68f4752f26385d89d4f3addf6fcd4",
    "rust/window/icon.rc": "a528a29ae9345d7aebf7a122a53a6f321f8f5f5ffcc008b3616c02f4880802a4",
    "rust/window/src/cef_backend.rs":
        "a3a50b3a250da2bd7c7a3af4d3fff270dc396647c82d1c05af222f4e37a342d5",
    "rust/window/src/helper.rs": "36c3da809fff1e70a871e48ae27e01f03bfdc9eeaa7abc11abc4bffebec2dc22",
    "rust/window/src/mac.rs": "16b7b2b2068a94ddb91d939048a42f8477dfb6530986fc8919d115767d74795f",
    "rust/window/src/main.rs": "9721e9a6c684dd2611b052a20f41f3166b2c250d323a219ed7eb7686d3dab9fb",
    "rust/window/src/wry_backend.rs":
        "6d22eb9793a3e400734b5226f323b8075494a5e9564012388b106f2282bee1ac",
    "src/harness/browser.ts": "5caddfb00f7197fbcb7c7fa6ff4d8c22a8c728fba70c0bf962e027f937ba70ac",
    "src/project/assets.ts": "6c14d42c3d2a2dcc8c7a31345c74e98e5ed3c95975d95623b636188d88dc6556",
    "src/project/command.ts": "41d5e4b94c03a47f5f49a87e171b51b0d7c3ce8e5d4ea498552644f592e6e725",
    "src/project/engine.ts": "3cc06a5771986cd6b1170e8950fade4c5337befed95b20052d6272332cd0a7e4",
    "src/project/generate.ts": "96259e5ca06e98b8ccfbb5d80bd830a7bc2c12f92b6b91d704ad8be723f7f88d",
    "src/project/host.ts": "8f0efc66919b5f1c54016caad50de78a65bc55c976f79761b6a814deb1bd7823",
    "src/project/manifest.ts": "9be3b99ddbd6c4ca1f70c3cb603ee2e8c49c1e7e2cf27331def7120f79332a8b",
    "src/project/vite.ts": "325be2bce5301f996dafb9cc5a5f0d816eef62f388dceaa3f1fceb1d8af740e0",
};

const hash = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
const trackedSuffixNeighbours = [
    "packages/shallot-cli/bin/cli.test.ts",
    "packages/shallot-cli/bin/verify.probes.ts",
    "packages/shallot-cli/bin/verify-blank.tier.ts",
    "packages/shallot-cli/rust/window/.gitignore",
];

type Fixture = {
    root: string;
    owner: string;
    project: string;
    tracked: string[];
    inputs: Record<string, string>;
    outputs: Record<string, string>;
};

function fixture(): Fixture {
    const root = mkdtempSync(join(tmpdir(), "shallot-install-reader-"));
    const owner = join(root, "packages/shallot-cli");
    const project = join(root, "project");
    const shipped = join(project, "node_modules/@dylanebert/shallot");
    const tracked = EXPECTED_INPUTS.filter((file) => file !== "../shallot/scripts/projections.ts")
        .map((file) => `packages/shallot-cli/${file}`)
        .concat(trackedSuffixNeighbours);
    mkdirSync(join(root, "packages/shallot/scripts"), { recursive: true });
    writeFileSync(join(root, "packages/shallot/scripts/projections.ts"), "projection\n");
    for (const file of tracked) {
        const relative = file.replace("packages/shallot-cli/", "");
        mkdirSync(dirname(join(owner, relative)), { recursive: true });
        writeFileSync(join(owner, relative), `source:${relative}\n`);
    }
    const inputs = Object.fromEntries(
        EXPECTED_INPUTS.map((file) => [file, hash(resolve(owner, file))]),
    );
    const outputs: Record<string, string> = {};
    for (const file of CANONICAL_OUTPUT_FILES) {
        const path = join(shipped, file);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, `output:${file}\n`);
        outputs[file] = hash(path);
    }
    writeFileSync(
        join(shipped, "dist/cli-inputs.json"),
        JSON.stringify({ inputs, outputs }, null, 2),
    );
    return { root, owner, project, tracked, inputs, outputs };
}

function rewriteRecord(fixture: Fixture): void {
    writeFileSync(
        join(fixture.project, "node_modules/@dylanebert/shallot/dist/cli-inputs.json"),
        JSON.stringify({ inputs: fixture.inputs, outputs: fixture.outputs }, null, 2),
    );
}

function runReader(fixture: Fixture): number {
    const driver = join(fixture.root, "reader.ts");
    writeFileSync(
        driver,
        `import { canonicalProjectionArm } from ${JSON.stringify(resolve(import.meta.dir, "harness.ts"))};\n` +
            `canonicalProjectionArm(${JSON.stringify(fixture.project)}, ${JSON.stringify(fixture.owner)}, ${JSON.stringify(fixture.tracked)});\n`,
    );
    const result = Bun.spawnSync(["bun", driver], {
        cwd: resolve(import.meta.dir, ".."),
        stdout: "pipe",
        stderr: "pipe",
    });
    return result.exitCode;
}

function expectReaderExit(mutate: (fixture: Fixture) => void, exit: number): void {
    const current = fixture();
    try {
        mutate(current);
        rewriteRecord(current);
        expect(runReader(current)).toBe(exit);
    } finally {
        rmSync(current.root, { recursive: true, force: true });
    }
}

test("the independent projection census pins 46 inputs, 48 outputs, and excluded suffix neighbours", () => {
    expect(EXPECTED_INPUTS).toHaveLength(46);
    expect(Object.keys(EXPECTED_OUTPUT_HASHES)).toHaveLength(48);
    expect(Object.keys(EXPECTED_OUTPUT_HASHES).sort()).toEqual(CANONICAL_OUTPUT_FILES);
    expect(
        Object.values(EXPECTED_OUTPUT_HASHES).every((value) => /^[0-9a-f]{64}$/.test(value)),
    ).toBe(true);

    const current = fixture();
    try {
        expect(
            canonicalInputFiles(current.tracked).concat("../shallot/scripts/projections.ts").sort(),
        ).toEqual(EXPECTED_INPUTS);
        expect(canonicalInputFiles(current.tracked)).not.toContain("bin/verify-blank.tier.ts");
        expect(canonicalInputFiles(current.tracked)).not.toContain("bin/cli.test.ts");
        expect(canonicalInputFiles(current.tracked)).not.toContain("bin/verify.probes.ts");
        expect(canonicalInputFiles(current.tracked)).not.toContain("rust/window/.gitignore");
        expect(canonicalInputFiles(current.tracked)).toContain("bin/verify.ts");
        expect(canonicalInputFiles(current.tracked)).toContain("rust/window/Cargo.toml");
    } finally {
        rmSync(current.root, { recursive: true, force: true });
    }
});

test("the admitted reader has a green subprocess baseline", () => {
    expectReaderExit(() => {}, 0);
});

test("omitted input, extra tier, and equal-size substitution fail membership and exit", () => {
    expectReaderExit((current) => {
        delete current.inputs["bin/cli.ts"];
    }, 1);
    expectReaderExit((current) => {
        current.inputs["bin/verify-blank.tier.ts"] = hash(
            join(current.owner, "bin/verify-blank.tier.ts"),
        );
    }, 1);
    expectReaderExit((current) => {
        delete current.inputs["bin/cli.ts"];
        current.inputs["bin/substituted.ts"] = "0".repeat(64);
    }, 1);
});

test("canonical and installed byte mutations fail their own hash predicates and exit", () => {
    expectReaderExit((current) => {
        writeFileSync(join(current.owner, "bin/cli.ts"), "changed source\n");
    }, 1);
    expectReaderExit((current) => {
        writeFileSync(
            join(current.project, "node_modules/@dylanebert/shallot/dist/vite.js"),
            "changed output\n",
        );
    }, 1);
});
