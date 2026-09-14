# Adding an engine

Implement `BaseEngine.run` and optional `close`, load resources in initialization,
and add a lazy import entry to `engines/registry.py`. Return `CountResult` for RGB input.
Add meaningful contract tests and a JSON-formatted YAML parameter file under
`configs/benchmark`. Add the name to `default.yaml` only when ready for default use.

Hough, regression, and vision LLM modules are placeholders. Model-file checksums and
remote model revisions must be included in benchmark identity when these are implemented.
