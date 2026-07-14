#!/usr/bin/env python3

"""Build chronological SWE-bench Pro issue chains grouped by repository.

Run from the repository root:

    uv run python scripts/swebench/analyze_swebench_pro_chains.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

from simple_agent_lab.llm.env import (
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    load_dotenv,
)

ROOT = Path(__file__).resolve().parents[2]
WORK_DATA_DIR = ROOT / "datasets" / "swebench_pro"
WORK_CACHE_DIR = WORK_DATA_DIR / "cache"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"

DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
PATH_TOKEN_RE = re.compile(r"`?((?:[\w@.+-]+/)+[\w@.+-]+\.[\w.+-]+)`?")
NO_INTERFACE_RE = re.compile(
    r"\bno (?:new )?(?:public )?interfaces? (?:are )?(?:introduced|changed|added)\b",
    re.I,
)
LANGUAGE_EXTENSIONS = {
    "go": (".go",),
    "js": (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"),
    "python": (".py", ".pyi"),
    "ts": (".ts", ".tsx"),
}
NOISE_PARTS = {
    "__snapshots__",
    "__tests__",
    "changelog",
    "cypress",
    "doc",
    "docs",
    "documentation",
    "e2e",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "i18n",
    "language",
    "locale",
    "locales",
    "spec",
    "specs",
    "test",
    "testdata",
    "testing",
    "tests",
    "translation",
    "translations",
}
NOISE_NAMES = {
    "license",
    "license.md",
    "package-lock.json",
    "pnpm-lock.yaml",
    "readme.md",
    "yarn.lock",
}
NOISE_SUFFIXES = (
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".lock",
    ".md",
    ".png",
    ".rst",
    ".snap",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)


@dataclass(frozen=True)
class Issue:
    repo: str
    instance_id: str
    base_commit: str
    commit_time: datetime
    files: frozenset[str]
    raw_files: frozenset[str]
    files_source: str


@dataclass
class LlmPatchFileSelector:
    model: str
    cache_path: Path
    api_key_env: str
    base_url: str
    max_patch_chars: int
    max_workers: int

    def __post_init__(self) -> None:
        self.cache: dict[str, list[str]] = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.is_file()
            else {}
        )

    def select_many(
        self, items: list[dict], raw_files_by_instance: dict[str, frozenset[str]]
    ) -> dict[str, frozenset[str]]:
        selected: dict[str, frozenset[str]] = {}
        missing: list[tuple[str, dict, frozenset[str]]] = []
        for item in items:
            files = raw_files_by_instance[item["instance_id"]]
            key = self.cache_key(item, files)
            if key in self.cache:
                selected[item["instance_id"]] = self.cached_files(key, files)
            else:
                missing.append((key, item, files))

        if missing:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.fetch, item, files): (
                        key,
                        item["instance_id"],
                        files,
                    )
                    for key, item, files in missing
                }
                completed = 0
                for future in as_completed(futures):
                    key, instance_id, files = futures[future]
                    self.cache[key] = sorted(future.result())
                    selected[instance_id] = self.cached_files(key, files)
                    completed += 1
                    self.cache_path.write_text(
                        json.dumps(dict(sorted(self.cache.items())), indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"llm_file_filter completed={completed}/{len(futures)} cached_total={len(selected)}/{len(items)}"
                    )
        return selected

    def cache_key(self, item: dict, files: frozenset[str]) -> str:
        return sha256(
            json.dumps(
                {
                    "model": self.model,
                    "instance_id": item["instance_id"],
                    "files": sorted(files),
                    "patch_sha": sha256(item["patch"].encode()).hexdigest(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def cached_files(self, key: str, files: frozenset[str]) -> frozenset[str]:
        return frozenset(file for file in self.cache[key] if file in files) or files

    def fetch(self, item: dict, files: frozenset[str]) -> frozenset[str]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"{self.api_key_env} is required when --llm-noise-filter is enabled"
            )
        payload = {
            "model": self.model,
            "reasoning_effort": "high",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Select the patch files that are semantically central to solving the issue. "
                        "Drop tests, docs, generated files, fixtures, translations, snapshots, lockfiles, and formatting-only files. "
                        'Return only JSON like {"files":["path/to/file.py"]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "repo": item["repo"],
                            "language": item.get("repo_language", ""),
                            "instance_id": item["instance_id"],
                            "problem_statement": item.get("problem_statement", "")[
                                :4000
                            ],
                            "candidate_files": sorted(files),
                            "patch": item["patch"][: self.max_patch_chars],
                        }
                    ),
                },
            ],
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            content = json.loads(response.read().decode())["choices"][0]["message"][
                "content"
            ]
        return frozenset(
            json.loads(content[content.index("{") : content.rindex("}") + 1])["files"]
        )


@dataclass
class CommitTimeResolver:
    cache_path: Path
    token_env: str
    offline: bool

    def __post_init__(self) -> None:
        self.cache: dict[str, str] = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.is_file()
            else {}
        )

    def get(self, repo: str, sha: str) -> datetime:
        key = f"{repo}@{sha}"
        if key not in self.cache:
            if self.offline:
                raise ValueError(f"missing commit time cache entry for {key}")
            self.cache[key] = self.fetch(repo, sha)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(dict(sorted(self.cache.items())), indent=2) + "\n",
                encoding="utf-8",
            )
        return parse_time(self.cache[key])

    def fetch(self, repo: str, sha: str) -> str:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token := os.getenv(self.token_env):
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"https://api.github.com/repos/{repo}/commits/{sha}", headers=headers
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())["commit"]["committer"]["date"]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def patch_files(patch: str) -> frozenset[str]:
    return frozenset(match.group(2) for match in DIFF_FILE_RE.finditer(patch))


def interface_files(interface: str) -> frozenset[str]:
    interface = interface.strip().strip('"')
    if not interface or NO_INTERFACE_RE.search(interface):
        return frozenset()
    return frozenset(
        normalize_file(match.group(1)) for match in PATH_TOKEN_RE.finditer(interface)
    )


def normalize_file(file: str) -> str:
    file = file.strip().strip("`'\".,;:)")
    if file.startswith(("a/", "b/")):
        file = file[2:]
    return file


def relation_files(item: dict) -> tuple[frozenset[str], str]:
    files = interface_files(item.get("interface", ""))
    if files:
        return files, "interface"
    return patch_files(item["patch"]), "patch"


def filter_relation_files(files: frozenset[str], language: str) -> frozenset[str]:
    filtered = frozenset(file for file in files if not is_noise_file(file, language))
    return filtered or files


def is_noise_file(file: str, language: str) -> bool:
    lower = file.lower()
    parts = lower.split("/")
    name = parts[-1]
    if any(part in NOISE_PARTS for part in parts) or name in NOISE_NAMES:
        return True
    if is_test_file(name) or is_generated_file(name):
        return True
    if language in LANGUAGE_EXTENSIONS and not name.endswith(
        LANGUAGE_EXTENSIONS[language]
    ):
        return True
    return lower.endswith(NOISE_SUFFIXES)


def is_test_file(name: str) -> bool:
    return (
        name.startswith("test_")
        or name.endswith("_test.go")
        or name.endswith("_test.py")
        or ".spec." in name
        or ".test." in name
    )


def is_generated_file(name: str) -> bool:
    return name.endswith((".generated.go", ".gen.go", ".pb.go", ".min.js"))


def load_issues(
    data_path: Path,
    resolver: CommitTimeResolver,
    filter_noise: bool,
    llm_selector: LlmPatchFileSelector | None,
) -> list[Issue]:
    issues = []
    items = json.loads(data_path.read_text(encoding="utf-8"))
    relations = {item["instance_id"]: relation_files(item) for item in items}
    llm_filtered = (
        llm_selector.select_many(
            [item for item in items if relations[item["instance_id"]][1] == "patch"],
            {item["instance_id"]: relations[item["instance_id"]][0] for item in items},
        )
        if llm_selector
        else {}
    )
    for item in items:
        raw_files, files_source = relation_files(item)
        files = llm_filtered.get(item["instance_id"], raw_files)
        if filter_noise:
            files = filter_relation_files(files, item.get("repo_language", ""))
        issues.append(
            Issue(
                repo=item["repo"],
                instance_id=item["instance_id"],
                base_commit=item["base_commit"],
                commit_time=resolver.get(item["repo"], item["base_commit"]),
                files=files,
                raw_files=raw_files,
                files_source=files_source,
            )
        )
    return issues


def issue_chains(
    issues: list[Issue], min_chain_size: int, ignore_file_freq: int = 0
) -> list[dict]:
    file_freq = Counter(file for issue in issues for file in issue.files)
    union_find = UnionFind(len(issues))
    first_issue_by_file: dict[str, int] = {}
    for idx, issue in enumerate(issues):
        for file in issue.files:
            if ignore_file_freq and file_freq[file] > ignore_file_freq:
                continue
            if file in first_issue_by_file:
                union_find.union(first_issue_by_file[file], idx)
            else:
                first_issue_by_file[file] = idx

    groups: dict[int, list[Issue]] = defaultdict(list)
    for idx, issue in enumerate(issues):
        groups[union_find.find(idx)].append(issue)

    chains = []
    for chain_idx, chain_issues in enumerate(
        sorted(
            (
                sorted(group, key=lambda issue: (issue.commit_time, issue.instance_id))
                for group in groups.values()
            ),
            key=lambda group: (group[0].commit_time, group[0].instance_id),
        ),
        start=1,
    ):
        if len(chain_issues) < min_chain_size:
            continue
        file_counts = Counter(file for issue in chain_issues for file in issue.files)
        chains.append(
            {
                "chain_id": f"{chain_issues[0].repo.replace('/', '__')}-{chain_idx:04d}",
                "issue_count": len(chain_issues),
                "shared_files": sorted(
                    file for file, count in file_counts.items() if count > 1
                ),
                "files": sorted(file_counts),
                "issues": [
                    {
                        "instance_id": issue.instance_id,
                        "base_commit": issue.base_commit,
                        "commit_time": issue.commit_time.isoformat(),
                        "files_source": issue.files_source,
                        "files": sorted(issue.files),
                        "raw_files": sorted(issue.raw_files),
                    }
                    for issue in chain_issues
                ],
            }
        )
    return chains


def analyze(
    issues: list[Issue], min_chain_size: int, ignore_file_freq: int = 0
) -> dict:
    repos = []
    for repo, repo_issues in sorted(group_by_repo(issues).items()):
        sorted_issues = sorted(
            repo_issues, key=lambda issue: (issue.commit_time, issue.instance_id)
        )
        repos.append(
            {
                "repo": repo,
                "issue_count": len(sorted_issues),
                "first_commit_time": sorted_issues[0].commit_time.isoformat(),
                "last_commit_time": sorted_issues[-1].commit_time.isoformat(),
                "chains": issue_chains(sorted_issues, min_chain_size, ignore_file_freq),
            }
        )
    return {"repo_count": len(repos), "issue_count": len(issues), "repos": repos}


def group_by_repo(issues: list[Issue]) -> dict[str, list[Issue]]:
    grouped: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        grouped[issue.repo].append(issue)
    return grouped


def print_summary(result: dict) -> None:
    print(f"repos={result['repo_count']} issues={result['issue_count']}")
    for repo in result["repos"]:
        chain_count = len(repo["chains"])
        issue_count = sum(chain["issue_count"] for chain in repo["chains"])
        longest = max((chain["issue_count"] for chain in repo["chains"]), default=0)
        print(
            f"{repo['repo']}: issues={repo['issue_count']} related_chains={chain_count} "
            f"chained_issues={issue_count} longest={longest}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=ROOT / ".env",
        help="Load provider credentials from this dotenv file without overriding exports.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=WORK_DATA_DIR / "swe_bench_pro.json",
        help="Local JSON export of the ScaleAI/SWE-bench_Pro test split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_issue_chains.json",
        help="Where to write the chain analysis JSON.",
    )
    parser.add_argument(
        "--commit-cache",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_commit_times.json",
        help="Commit timestamp cache JSON.",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing a GitHub token.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use commit timestamps already present in the cache.",
    )
    parser.add_argument(
        "--min-chain-size",
        type=int,
        default=4,
        help="Minimum issue count required for a chain to be included.",
    )
    parser.add_argument(
        "--filter-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter tests/docs/generated/config files before matching.",
    )
    parser.add_argument(
        "--llm-noise-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an LLM to select central files when interface files are absent.",
    )
    parser.add_argument(
        "--llm-noise-model",
        default="gpt-5.4-mini-2026-03-17",
        help="OpenAI-compatible model used for patch-file selection.",
    )
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=WORK_CACHE_DIR / "swe_bench_pro_llm_file_filter.json",
        help="Cache for LLM patch-file selections.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=OPENAI_AUTH_ENV,
        help="Environment variable containing the LLM bearer token.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible API base URL (defaults to OPENAI_BASE_URL, then api.openai.com).",
    )
    parser.add_argument(
        "--max-patch-chars",
        type=int,
        default=12_000,
        help="Maximum patch characters sent per LLM filtering request.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=50,
        help="Maximum concurrent LLM file-filtering requests.",
    )
    parser.add_argument(
        "--ignore-file-freq",
        type=int,
        default=0,
        help="Ignore files touched by more than N issues when linking; 0 disables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv(args.dotenv)
    llm_base_url = (
        args.llm_base_url
        or os.getenv(OPENAI_BASE_URL_ENV, "").strip()
        or DEFAULT_LLM_BASE_URL
    )
    result = analyze(
        load_issues(
            args.data_path,
            CommitTimeResolver(args.commit_cache, args.token_env, args.offline),
            args.filter_noise,
            LlmPatchFileSelector(
                args.llm_noise_model,
                args.llm_cache,
                args.llm_api_key_env,
                llm_base_url,
                args.max_patch_chars,
                args.llm_workers,
            )
            if args.llm_noise_filter
            else None,
        ),
        args.min_chain_size,
        args.ignore_file_freq,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(result)


if __name__ == "__main__":
    main()
