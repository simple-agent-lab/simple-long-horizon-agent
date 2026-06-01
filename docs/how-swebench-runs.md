# How a full SWE-bench run works

A human-facing walkthrough of what actually happens, end to end, when you run
the whole SWE-bench benchmark on this framework — from "a dataset of bug reports"
to "a resolved-rate score." It favors the story and the *why* over API details;
for those, see [ADR 0017](decisions/0017-generic-containerized-eval-framework.md),
[`evals/README.md`](../evals/README.md), and the step-by-step
[integration guide](agent-native/integrating-a-docker-eval-suite.md).

## The big picture: three separate stages

A benchmark run is **not one monolithic thing**. It is three stages that are
deliberately kept apart, because each can change independently:

```
 ① PREPARE            ② GENERATE                      ③ SCORE
 ─────────            ──────────                      ───────
 build/pull images    run the agent in each image     official SWE-bench harness
 + load the dataset   → produce a patch per instance   applies each patch in a
                      → collect prediction.jsonl        clean container, runs the
                                                        tests → resolved / not
```

- **Stage ② (generate)** is what this framework owns: drive an agent inside each
  instance's container and collect the patch it produces.
- **Stage ③ (score)** is the *official* SWE-bench harness, run separately. Our
  framework never decides whether a patch is correct — it only produces patches.
  This separation is on purpose: the gold test data lives only in the scorer, and
  the raw agent trajectories stay reusable even if scoring rules change later.

The rest of this doc zooms into each stage.

## Why SWE-bench needs a container per instance

SWE-bench is not "one environment." Each **instance** is a real GitHub bug: a
specific repo, checked out at the commit just before the fix, with that repo's
exact dependencies installed. So SWE-bench Verified's 500 instances are 500
different Docker images. There is no shared "SWE-bench environment" to download —
you build them (or, for SWE-bench Pro, pull them from Docker Hub).

That is the core reason the framework is container-shaped: the agent has to work
inside the *real* broken repo, with its real toolchain, exactly as a developer
would. `/testbed` in the container is that repo.

## A key idea: the agent code is injected at runtime, not baked in

We do **not** build our agent into those 500 images. When a container starts, its
first job (a small bootstrap script) is to `pip install simple-agent-lab` into an
isolated virtualenv, then run the generic agent runner. The benchmark image is
used untouched.

Why: agent code changes constantly; the benchmark images are huge and stable.
Baking the agent in would mean rebuilding hundreds of multi-GB images on every
agent tweak. Runtime injection decouples the two — and lets a multi-machine run
use the *upstream* images as-is.

## Stage ②, zoomed in: one instance, start to finish

This is the heart of it. When the framework runs a single instance
(`run_suite_instance`), here is the sequence. The host orchestrates but does no
heavy compute — the agent runs *inside* the container and talks to the model API
directly:

```
 HOST (orchestrator)              STORE (shared dir)        CONTAINER (the bug's image)
 ───────────────────              ─────────────────         ───────────────────────────
 1. work out how to launch
    this instance (image,
    workdir, capabilities)
 2. make the run directory
 3. strip the gold answer
    from the instance record
 4. write instance.json ─────────▶ input/instance.json
 5. start the container ──────────────────────────────────▶ 6. bootstrap: find python,
    (then just wait)                                            pip install the agent wheel
                                                             7. read instance.json
                                                             8. PREPARE: snapshot a clean
                                                                git baseline of /testbed
                                                             9. build the task from the
                                                                bug's problem statement
                                                            10. ★ RUN THE AGENT LOOP ★
                                                                repeat:
                                                                  - call the model
                                                                  - run bash in /testbed
                                                                    (read code, edit, test)
                                   trajectory.jsonl ◀──────────  - every ~2s: write the
                                   (live, updates)                 trajectory so a viewer
                                                                   can watch in real time
                                                            11. EXTRACT: git diff →
                                   out/result.json ◀──────────    {"model_patch": "..."}
12. container exits; host
    reads exit code + logs,
    removes the container
13. shape the scorer row:
    result.json → prediction.jsonl
14. done: artifacts on disk
```

The takeaways a person should leave with:

- **The container is self-contained.** Once started, it calls the model API and
  edits the repo on its own. If the host process died here, the container would
  keep running. (That property is what makes long, host-detached runs possible —
  see "Variations" below.)
- **Two products per instance.** `result.json` (the raw patch the agent
  produced) and `trajectory.jsonl` (the full record of how it got there, written
  live). The host turns the first into the scorer-facing `prediction.jsonl`.
- **The host is light.** It launches, waits, and tidies up. All the expensive
  work — the agent's reasoning and the bug-fixing — is inside the container.

## Stage ②, zoomed out: the whole dataset at once

Running 500 instances one-at-a-time would take forever, so the framework runs
many concurrently. `run_dataset` is a small controller: it calls the
single-instance flow above once per instance on a thread pool.

```
 load_instances("test")              ← you provide these (HuggingFace, JSONL, …)
        │  a stream of instance dicts
        ▼
 run_dataset(..., concurrency=N)
        │
   thread pool of size N
        ├─ instance 1 ─→ [the full single-instance flow] ─→ its own container
        ├─ instance 2 ─→ [ … ]                            ─→ its own container
        ├─ …                                                 (N at a time)
        └─ instance K ─→ [ … ]
        │   a failed run (infra hiccup) is retried up to max_attempts
        ▼
   every artifact lands under one run_root:
     <run_root>/<run_id>/<instance_id>/out/{trajectory,prediction}.jsonl
        ▼
   DatasetReport — how many ok / failed
```

Each instance gets its own container and its own sub-directory, so concurrent
runs never collide, and every `prediction.jsonl` ends up under one tree ready for
scoring. This is the same "pool of workers" shape that large distributed eval
frameworks use — minus all the machinery (GPU placement, weight sync) that
training needs and evaluation does not.

## Stage ③: scoring (separate, official)

`run_dataset` finishes with a `prediction.jsonl` per instance — each just a
patch. **It says nothing about whether the patch is correct.** Correctness is
decided by the official SWE-bench harness, run as its own step: for each
prediction it spins up a *clean* container, applies the patch, runs the bug's
`FAIL_TO_PASS` and `PASS_TO_PASS` tests, and marks the instance resolved or not.
The final score is the resolved rate.

That step uses the gold test data — which is exactly why it is kept out of the
generation path. The agent never sees the answer; the scorer never runs the agent.

## Putting it together

```
 ① build/pull the per-instance images   +   load_instances()
        │
 ② run_dataset → N containers in parallel
        each container: bootstrap → prepare → agent loop → git diff → result.json
        host aggregates → prediction.jsonl for the whole dataset
        (a trace viewer can tail trajectory.jsonl live)
        │
 ③ official SWE-bench harness reads prediction.jsonl
        → clean container per patch → run tests → resolved rate
```

## Variations on stage ②

The single-instance flow is fixed; *where* it runs and *whether the host waits*
are swappable, without changing the suite or the agent:

- **Local development, no Docker.** Swap the container backend for the in-process
  one and the same agent code runs in your Python process against a local
  workspace — fast to iterate and debug, no image build.
- **Long runs the host shouldn't babysit.** Instead of one blocking call, submit
  all the containers, write a manifest, and let the host leave; a fresh process
  later *reconciles* the batch — re-reads the manifest, waits for each container,
  and collects results. The containers ran detached the whole time.
- **Many machines.** Point the backend at remote Docker daemons; each worker runs
  containers and the host pulls the artifacts back to its one `run_root`. Workers
  run nothing of ours but a Docker daemon and the images.

All three are "change one argument," because the launch location and the byte
movement are separate, swappable seams. The deployment details live in
[multi-machine-deployment.md](agent-native/multi-machine-deployment.md).
