# Adding an engine

Implement `BaseEngine.run` and optional `close`, load resources in initialization,
and add a lazy import entry to `engines/registry.py`. Return `CountResult` for RGB input.
Add meaningful contract tests and a JSON-formatted YAML parameter file under
`configs/benchmark`. Add the name to `default.yaml` only when ready for default use.

The vision LLM module is a placeholder. Engines that take a `model_path` parameter get the
weights' SHA-256 added to the benchmark manifest automatically (see `regression`); remote
model revisions must be included in benchmark identity, as `grounding_dino` does by recording
the resolved Hugging Face commit hash and transformers version in the manifest.
