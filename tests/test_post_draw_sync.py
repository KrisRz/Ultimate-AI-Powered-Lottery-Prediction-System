"""The post-draw sync must never hand half-merged data to the model.

Reproduces the failure of 2026-09-05: the Mac fetches a draw into the tracked
CSVs 15 min before the collector commits the same draw, so the local tree is
always dirty; miss a run or two and `git pull --rebase --autostash` conflicts
on the autostash, leaving `data/*.csv` full of `<<<<<<<` markers that the EV
model then reads as "jackpot £2M, rollover 0 of 5".
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SYNC_SCRIPT = Path("scripts/monitoring/sync_collector_data.sh")
TIERS = "data/prize_tiers.csv"
HEADER = "draw_number,draw_date,tier,winners\n"

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], env=GIT_ENV,
                          capture_output=True, text=True, check=True).stdout


def _draw_row(draw: int) -> str:
    return f"{draw},2026-09-{draw - 3200:02d},1,{draw * 7}\n"


def _write_draws(repo: Path, through: int) -> None:
    rows = HEADER + "".join(_draw_row(d) for d in range(3201, through + 1))
    (repo / TIERS).write_text(rows)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


@pytest.fixture()
def repos(tmp_path):
    """An upstream the collector commits to, and the local clone beside it."""
    upstream = tmp_path / "upstream"
    (upstream / "data").mkdir(parents=True)
    (upstream / "scripts" / "monitoring").mkdir(parents=True)
    shutil.copy(SYNC_SCRIPT, upstream / SYNC_SCRIPT)
    _git(upstream.parent, "init", "-b", "main", str(upstream))
    _write_draws(upstream, 3203)
    _commit(upstream, "data: collect draw 3203")

    local = tmp_path / "local"
    _git(tmp_path, "clone", str(upstream), str(local))
    return upstream, local


def _run_sync(local: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(local / SYNC_SCRIPT)], env=GIT_ENV,
                          capture_output=True, text=True)


def test_local_fetch_is_discarded_and_the_collector_wins(repos):
    """The exact 2026-09-05 shape: local +1 draw, remote +3, same file end."""
    upstream, local = repos
    _write_draws(local, 3204)                      # the Mac's 21:30 UTC fetch
    for draw in (3204, 3205, 3206):                # three collector commits
        _write_draws(upstream, draw)
        _commit(upstream, f"data: collect draw {draw}")

    result = _run_sync(local)

    assert result.returncode == 0, result.stderr
    synced = (local / TIERS).read_text()
    assert "<<<<<<<" not in synced
    assert synced == (upstream / TIERS).read_text()
    assert "3206" in synced
    assert _git(local, "status", "--porcelain") == ""


def test_untracked_local_files_survive_the_sync(repos):
    """The ledger and logs are local-only - the sync must not sweep them away."""
    upstream, local = repos
    (local / "data" / "ledger.csv").write_text("draw,stake\n3196,20\n")
    _write_draws(upstream, 3204)
    _commit(upstream, "data: collect draw 3204")

    assert _run_sync(local).returncode == 0
    assert (local / "data" / "ledger.csv").read_text() == "draw,stake\n3196,20\n"


def test_refuses_to_run_on_an_unmerged_tree(repos):
    """Whatever left `UU` files behind, running the model on them is worse."""
    upstream, local = repos
    _write_draws(upstream, 3204)
    _commit(upstream, "data: collect draw 3204")
    (local / TIERS).write_text(HEADER + _draw_row(3204).replace(",1,", ",9,"))
    _commit(local, "local edit that conflicts")
    _git(local, "fetch", "origin", "main")
    subprocess.run(["git", "-C", str(local), "merge", "origin/main"],
                   env=GIT_ENV, capture_output=True, text=True)  # conflicts
    assert _git(local, "ls-files", "-u") != ""

    result = _run_sync(local)

    assert result.returncode == 1
    assert "refusing" in result.stdout
    assert TIERS in result.stdout


def test_diverged_branch_keeps_intact_data_and_says_so(repos):
    """A local commit is not corruption - warn, keep the tree, let the run go on."""
    upstream, local = repos
    _write_draws(local, 3204)
    _commit(local, "a local commit the collector never saw")
    _write_draws(upstream, 3205)
    _commit(upstream, "data: collect draw 3205")

    result = _run_sync(local)

    assert result.returncode == 0
    assert "cannot fast-forward" in result.stdout
    assert "<<<<<<<" not in (local / TIERS).read_text()
