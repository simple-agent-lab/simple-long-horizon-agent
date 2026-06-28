# Shared: ensure a Linux uv binary for the SWE-bench container; sets SWEBENCH_UV_BIN.
#
# The container builds a Python 3.11 venv with uv, but the host `uv` is the wrong
# OS/arch on macOS and can't exec inside the x86_64 Linux image. Fetch a static
# Linux uv once (cached under evals/out/uv-linux) and bind-mount it at /tmp/uv.
# Override the target with UV_LINUX_TARGET (e.g. uv-aarch64-unknown-linux-gnu).
swebench_ensure_linux_uv() {
  local target="${UV_LINUX_TARGET:-uv-x86_64-unknown-linux-musl}"
  local dir="evals/out/uv-linux"
  SWEBENCH_UV_BIN="$dir/$target/uv"
  if [ ! -x "$SWEBENCH_UV_BIN" ]; then
    echo "==> Fetching Linux uv ($target) for the container..." >&2
    mkdir -p "$dir"
    curl -fsSL \
      "https://github.com/astral-sh/uv/releases/latest/download/${target}.tar.gz" \
      -o "$dir/uv.tgz"
    tar xzf "$dir/uv.tgz" -C "$dir"
    chmod +x "$SWEBENCH_UV_BIN"
  fi
}
