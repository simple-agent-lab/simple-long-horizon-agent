# Examples

The current runnable demo is the tiny message runtime:

```bash
python3 scripts/run_tiny_demo.py --recipe all
```

It uses deterministic toy agents instead of real LLM calls. This keeps the
architecture easy to inspect. To connect a real model later, replace an agent's
`act` function with a model call while keeping the same runtime shape.

The core idea is:

```text
Agent + Message + State + context_view() + run()
```

`context_view()` is the only context management boundary. Try limiting each
agent's visible history:

```bash
python3 scripts/run_tiny_demo.py --recipe debate --last-messages 1
```

To call a real model later, pass visible messages through `model_messages(...)`:

```python
from simple_agent_lab import context_view, model_messages

visible = context_view(agent, state)
payload = model_messages(visible)
```

## Run

Run the current smoke demo:

```bash
bash runs/run_examples.sh
```

Each script also has a small CLI:

```bash
python3 scripts/run_tiny_demo.py --recipe debate --task "Compare debate and pipeline agents" --no-trace
```

## Cases

- `../scripts/run_tiny_demo.py`: the smallest message-runtime demo; this is the preferred core direction.
- `design_versions/`: three runnable core-loop sketches: functional loop, mailbox scheduler, and event runtime.
