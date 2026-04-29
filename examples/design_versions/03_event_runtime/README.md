# 03 Event Runtime

This version is shaped for observability and multiple model providers.

Files:

- `core.py`: `RuntimeState`, `RuntimeEvent`, `AgentLoop`, and message conversion.
- `models.py`: `ModelConfig`, `ModelClient`, `FakeModel`,
  `OpenAIResponsesClient`, and `load_model`.
- `demo.py`: runnable with fake model by default.

The loop is:

```text
context_view -> model_request event -> ModelClient.generate
  -> model_response event -> message event -> stop event
```

Run with no network:

```bash
python3 examples/design_versions/03_event_runtime/demo.py
```

Run later with OpenAI Responses:

```bash
python3 examples/design_versions/03_event_runtime/demo.py --provider openai --model gpt-5-mini
```

The core still does not store provider response objects. Provider details are
converted back into `Message.data`.
