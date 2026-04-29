# 01 Functional Loop

This is the smallest version.

Everything lives in `demo.py`:

- `Message`
- `context_view`
- `fake_model`
- `run_loop`

There is no `State`, `Agent`, `Event`, or runtime class. The message list is
the trace.

The loop is:

```text
context_view(messages) -> fake_model -> append message -> stop or continue
```

Run:

```bash
python3 examples/design_versions/01_functional_loop/demo.py
```
