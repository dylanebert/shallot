# Default-look scene captures

`scenes.ts` is the fixed scene set. `palette.ts` is its shared material-colour table; adding a palette renders the same scene list under a new `captures/<palette>/` directory.

Run the capture suite from the repository root:

```sh
bunx playwright test --config diagnostics/default-look/playwright.config.ts
```

The suite requires a browser WebGPU hardware adapter, sets a 1280×720 viewport at device scale 1, and writes one PNG per scene and palette to the ignored `captures/` directory. Software WebGPU adapters are refused. To verify repeatability on the same machine, save the PNG hashes, rerun the suite, and compare:

```sh
shasum -a 256 diagnostics/default-look/captures/current/*.png > /tmp/default-look.sha256
bunx playwright test --config diagnostics/default-look/playwright.config.ts
shasum -a 256 diagnostics/default-look/captures/current/*.png | diff -u /tmp/default-look.sha256 -
```
