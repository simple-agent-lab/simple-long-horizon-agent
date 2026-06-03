/**
 * GatewayClient — spawns the Python gateway and frames newline-delimited
 * JSON-RPC 2.0 over its stdio.
 *
 * The one architectural rule that matters (learned the hard way in the Python
 * stand-in): the streaming ack of `prompt.submit` (`{status:"streaming"}`) and
 * the turn's events arrive UNORDERED on the same stream. So we keep a single
 * read loop that dispatches every frame by shape — a frame whose `id` matches a
 * pending request resolves that promise; any `method:"event"` frame is fanned
 * out to event listeners. Requests and events never consume each other, so the
 * ack/event race simply cannot deadlock the UI.
 */

import { type ChildProcess, spawn } from "node:child_process";

export interface GatewayEvent {
  type: string;
  session_id?: string;
  payload: Record<string, unknown>;
}

export interface RpcError {
  code: number;
  message: string;
}

type Pending = { resolve: (value: unknown) => void; reject: (err: unknown) => void };

export interface GatewayOptions {
  /** Executable to spawn (e.g. "uv" or an explicit python path). */
  command: string;
  /** Args that launch the gateway module (e.g. ["run","python","-m",...]). */
  args: string[];
  /** Working directory; also used to set PYTHONPATH=<cwd>/src. */
  cwd: string;
}

export class GatewayClient {
  private proc: ChildProcess;
  private buf = "";
  private nextId = 0;
  private pending = new Map<string, Pending>();
  private eventListeners: Array<(event: GatewayEvent) => void> = [];
  private stderrListeners: Array<(line: string) => void> = [];
  private exitListeners: Array<(code: number | null) => void> = [];

  constructor(opts: GatewayOptions) {
    this.proc = spawn(opts.command, opts.args, {
      cwd: opts.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONPATH: `${opts.cwd}/src` },
    });
    const stdout = this.proc.stdout!;
    stdout.setEncoding("utf8");
    stdout.on("data", (chunk: string) => this.onStdout(chunk));

    const stderr = this.proc.stderr!;
    stderr.setEncoding("utf8");
    stderr.on("data", (chunk: string) => {
      for (const line of chunk.split("\n")) {
        if (line.trim()) this.stderrListeners.forEach((l) => l(line));
      }
    });

    this.proc.on("exit", (code) => {
      const err = new Error("gateway process exited");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      this.exitListeners.forEach((l) => l(code));
    });
  }

  onEvent(listener: (event: GatewayEvent) => void): void {
    this.eventListeners.push(listener);
  }

  onStderr(listener: (line: string) => void): void {
    this.stderrListeners.push(listener);
  }

  onExit(listener: (code: number | null) => void): void {
    this.exitListeners.push(listener);
  }

  /** Send a JSON-RPC request and resolve with its result (reject on error). */
  request<T = unknown>(method: string, params: Record<string, unknown>): Promise<T> {
    const id = `r${++this.nextId}`;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.write({ jsonrpc: "2.0", id, method, params });
    });
  }

  stop(): void {
    try {
      this.proc.stdin?.end();
    } catch {
      /* already closed */
    }
    this.proc.kill();
  }

  private onStdout(chunk: string): void {
    this.buf += chunk;
    let nl: number;
    while ((nl = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, nl).trim();
      this.buf = this.buf.slice(nl + 1);
      if (!line) continue;
      let frame: Record<string, unknown>;
      try {
        frame = JSON.parse(line);
      } catch {
        continue; // protocol noise; the gateway keeps stdout frames-only
      }
      this.dispatch(frame);
    }
  }

  private dispatch(frame: Record<string, unknown>): void {
    const id = frame.id as string | undefined;
    if (id != null && this.pending.has(id)) {
      const p = this.pending.get(id)!;
      this.pending.delete(id);
      if (frame.error) p.reject(frame.error as RpcError);
      else p.resolve(frame.result);
      return;
    }
    if (frame.method === "event") {
      const params = (frame.params ?? {}) as Record<string, unknown>;
      this.eventListeners.forEach((l) =>
        l({
          type: String(params.type ?? ""),
          session_id: params.session_id as string | undefined,
          payload: (params.payload ?? {}) as Record<string, unknown>,
        }),
      );
    }
  }

  private write(obj: unknown): void {
    this.proc.stdin?.write(`${JSON.stringify(obj)}\n`);
  }
}
