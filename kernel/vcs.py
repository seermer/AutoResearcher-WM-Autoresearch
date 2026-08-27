"""Version control: one git branch + worktree per node, for both repos."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PATHS


def _prefix() -> str:
    """Branch namespace, derived from the archive directory.

    A git branch can back only one worktree, so two archives sharing the `node/`
    namespace collide on their very first node. Namespacing by archive lets a test
    or a side experiment run against the same repos as a live archive.
    """
    name = PATHS.archive.name
    return "node" if name == "archive" else f"node-{name}"


NODE_BRANCH = _prefix() + "/{nid}"
TRASH_BRANCH = "trash-" + _prefix() + "/{nid}"


def git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {repo}: {p.stderr.strip()}")
    return p.stdout.strip()


def current_branch(repo: Path) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def branch_exists(repo: Path, name: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q", name],
                          capture_output=True).returncode == 0


def ensure_committed(repo: Path, message: str) -> str | None:
    """Commit everything in a worktree. Returns the sha, or None if nothing changed."""
    git(repo, "add", "-A")
    if not git(repo, "status", "--porcelain"):
        return None
    git(repo, "commit", "-q", "-m", message, "--no-verify")
    return git(repo, "rev-parse", "HEAD")


def diffstat(repo: Path, base: str = "HEAD~1") -> dict:
    if not branch_exists(repo, base):
        return {"files": [], "insertions": 0, "deletions": 0}
    files = [l for l in git(repo, "diff", "--name-only", base, "HEAD").splitlines() if l]
    nums = git(repo, "diff", "--numstat", base, "HEAD").splitlines()
    ins = dels = 0
    for line in nums:
        a, b, *_ = line.split("\t")
        ins += int(a) if a.isdigit() else 0
        dels += int(b) if b.isdigit() else 0
    return {"files": files, "insertions": ins, "deletions": dels}


def add_detached_worktree(repo: Path, commitish: str, path: Path) -> Path:
    """A read-only checkout at a commit. Detached so it never claims a branch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "prune", check=False)
    if path.exists():
        remove_worktree(repo, path)
    git(repo, "worktree", "add", "--detach", str(path), commitish)
    return path


def add_worktree(repo: Path, branch: str, base: str, path: Path) -> Path:
    """Materialize `branch` (created from `base`) as a worktree at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "prune", check=False)   # drop records of deleted checkouts
    if path.exists():
        remove_worktree(repo, path)
    if branch_exists(repo, branch):
        git(repo, "worktree", "add", str(path), branch)
    else:
        git(repo, "worktree", "add", "-b", branch, str(path), base)
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(path)],
                   capture_output=True)
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], capture_output=True)
    git(repo, "worktree", "prune", check=False)


def to_trash(repo: Path, nid: str) -> None:
    """Rename a failed node's branch out of the live namespace."""
    src, dst = NODE_BRANCH.format(nid=nid), TRASH_BRANCH.format(nid=nid)
    if branch_exists(repo, src):
        if branch_exists(repo, dst):
            git(repo, "branch", "-D", dst, check=False)
        git(repo, "branch", "-m", src, dst, check=False)


class NodeWorkspace:
    """The pair of worktrees (agents + sana) backing one node."""

    def __init__(self, nid: str, parent_nid: str | None):
        self.nid = nid
        self.branch = NODE_BRANCH.format(nid=nid)
        self.base = NODE_BRANCH.format(nid=parent_nid) if parent_nid else None
        self.root = PATHS.worktrees / nid

    @property
    def agents(self) -> Path:
        """Writable draft: the child's branch. edit_self writes here."""
        return self.root / "agents"

    @property
    def agents_frozen(self) -> Path:
        """Immutable snapshot of the PARENT's agent code. edit_self *runs* from here,
        so an agent rewriting its own package cannot corrupt the interpreter that is
        executing it mid-run (a half-written module, a truncated prompt file)."""
        return self.root / "agents_frozen"

    @property
    def sana(self) -> Path:
        return self.root / "sana"

    def create(self) -> "NodeWorkspace":
        agent_base = self.base if self.base and branch_exists(PATHS.repo, self.base) else \
            git(PATHS.repo, "rev-parse", "HEAD")
        sana_base = self.base if self.base and branch_exists(PATHS.sana, self.base) else \
            git(PATHS.sana, "rev-parse", "HEAD")
        add_worktree(PATHS.repo, self.branch, agent_base, self.agents)
        add_worktree(PATHS.sana, self.branch, sana_base, self.sana)
        add_detached_worktree(PATHS.repo, agent_base, self.agents_frozen)
        return self

    def release_frozen(self) -> None:
        """Drop the read-only snapshot once edit_self is done with it."""
        remove_worktree(PATHS.repo, self.agents_frozen)

    def destroy(self) -> None:
        remove_worktree(PATHS.repo, self.agents)
        remove_worktree(PATHS.repo, self.agents_frozen)
        remove_worktree(PATHS.sana, self.sana)
        if self.root.exists() and not any(self.root.iterdir()):
            self.root.rmdir()

    def trash(self) -> None:
        self.destroy()
        to_trash(PATHS.repo, self.nid)
        to_trash(PATHS.sana, self.nid)


def link_node_data(sana_worktree: Path, manifest: list[str]) -> list[str]:
    """Build a node's `data/` from its shard manifest.

    `data/` is gitignored, so each worktree starts empty. Shards are symlinked from
    the immutable store and are read-only on disk, so a node cannot write through to
    what its parent trained on. New data goes in the node-private `data/staging/`,
    which the kernel seals into the store only if the node succeeds.
    """
    from . import datastore
    return datastore.materialize(manifest, sana_worktree / "data")
