# Recorder CLI design

## Goal

Make `deafbench recorder` launch the existing Tk recorder after installing
`deafbench[recorder]`, even when the command is run outside a DeafBench source
checkout.

## User interface

```powershell
python -m pip install "deafbench[recorder]"
deafbench recorder
deafbench recorder --dataset non-speech-v1
```

`core-v1` is the default dataset. Existing recorder overrides remain available:
`--repo-root`, `--references`, and `--audio-dir`.

## Package layout

Move the recorder runtime into `deafbench.recorder` so it is included in the
wheel. Keep the old `tools.recorder` entry point as a thin compatibility wrapper
for repository workflows.

Bundle the current Core v1 and Non-speech v1 reference JSONL files as package
data. The recorder must not depend on the top-level `benchmarks/` directory
being present in the installed package.

## Workspace behavior

When no explicit `--references` path is supplied, treat `--repo-root` as the
workspace root. Its default is the current working directory for the installed
CLI.

The recorder uses:

```text
<workspace>/benchmarks/<dataset>/references.jsonl
<workspace>/benchmarks/<dataset>/audio/
```

If `references.jsonl` is missing for `core-v1` or `non-speech-v1`, copy the
bundled reference file into the workspace before opening the GUI. Never
overwrite an existing references file. A custom dataset still works when the
caller supplies an existing references file or already has one in the workspace.

## Dependency behavior

Keep NumPy and sounddevice in the existing `recorder` optional extra. Normal
commands such as `deafbench compare`, `deafbench report`, and `deafbench --help`
must not import recorder runtime dependencies. Import the GUI recorder only
after the `recorder` subcommand is selected.

## Error handling

Invalid dataset names keep the existing safe-name validation. Missing recorder
dependencies produce a direct install hint for `deafbench[recorder]`. Missing
custom references produce a clear file-not-found error before Tk opens.

## Testing

Use TDD. First add a failing CLI dispatch test and workspace/bootstrap tests.
Then move the recorder into the package, add packaged reference data, preserve
the legacy tools wrapper, and run the full Python 3.11 through 3.14 CI matrix.
