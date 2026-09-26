# Stage 1 artifact upload spike

- Run: https://github.com/dylanebert/shallot/actions/runs/36256868233
- Tested commit: `f69c70576ae0a5185dbc653ea8a2ee4877df8871`
- The upload and verification steps completed; the job ended failed by design because the failing fixture was retained as a failure, and the final verification step explicitly exited 1.
- Downloaded reports and child output are preserved in `failing/` and `passing/`. Both appeared under distinct `shallot-test-Linux-{invocation}-{run_id}` artifacts.
- The empty-selection invocation ran `--integration --base HEAD --diff HEAD --requires gpu --requires browser --no-unit-fallback`. It exited successfully, `.artifacts/` was empty after clearing, upload warned that no files existed, and there was no empty-selection artifact. Thus it did not upload the previous passing invocation's evidence.
- Input correction: the first spike invocation used only `--base HEAD --diff HEAD`; its downloaded report showed 16 tests, not an empty selection. The corrected invocation adds both requirement filters, which select zero rows. No source/runner behavior was altered.
- The temporary `artifact-spike.yml` workflow was only for this Actions run and is removed from the candidate.
