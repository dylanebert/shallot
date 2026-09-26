# Stage 1 artifact upload spike evidence

## Corrected proof

- Run: https://github.com/dylanebert/shallot/actions/runs/36257199121
- Exact tested commit: `b9ecf2a8d58211a52623bd6f3e8d7b9fb7ec28e3`
- The failing fixture had no `continue-on-error`; its step concluded `failure`. The passing invocation, empty selection, all three upload steps, download and final verification concluded `success`. The job concluded `failure` solely from the failing fixture; verification did not force a red.
- The verifier checked both downloaded JUnit reports and output under distinct failing and passing artifact names. Those downloaded files are preserved in `corrected-proof/failing/` and `corrected-proof/passing/`.
- Empty selection ran `--integration --base HEAD --diff HEAD --requires gpu --requires browser --no-unit-fallback`. It exited successfully after the preceding passing artifact was present; the upload warned that `.artifacts/` had no files, and no empty-selection artifact was downloaded.
- The temporary spike workflow was removed after this run. Production workflow and runner behavior were unchanged by the spike.

## Earlier insufficient run and input corrections

- Earlier run: https://github.com/dylanebert/shallot/actions/runs/36256868233, tested commit `f69c70576ae0a5185dbc653ea8a2ee4877df8871`. Its failing fixture used `continue-on-error` and final verification forced exit 1, so its job red did not prove original-failure preservation. Its downloaded reports are retained in `failing/` and `passing/` for traceability only, not as proof.
- Its initial empty-selection arguments selected 16 tests. The corrected filter pair above selected zero rows. This was an input correction, not a runner change.
- Workflow dispatch initially failed because GitHub only recognized workflows on the default branch. The spike was changed to a push-triggered workflow on an owned disposable branch. The corrected proof ran once; no unchanged retries were made.
