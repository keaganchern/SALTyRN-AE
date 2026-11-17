"""A reused Zephyr build slot must rebuild with the new kernel's harness: the installed files get a fresh mtime
(shutil.copy2 kept the git checkout's old one, so the incremental build kept the previous kernel's program)."""
import os
import time

from src.workflow.build import compiler


def test_installed_harness_and_kernel_are_newer_than_a_previous_build(tmp_path):
    harness_dir, app = tmp_path / "harness", tmp_path / "app"
    (harness_dir).mkdir()
    (app / "src").mkdir(parents=True)
    old = time.time() - 86400                      # as if checked out yesterday
    for f in (harness_dir / "k.c", tmp_path / "k-128.c"):
        f.write_text("int x;\n")
        os.utime(f, (old, old))
    built = time.time() - 60                       # the previous kernel's object, a minute ago
    assert compiler._install_harness("k", harness_dir, app)
    assert compiler._install_kernel("k", tmp_path / "k-128.c", app)
    assert (app / "src/main.c").stat().st_mtime > built
    assert (app / "kernel/k.c").stat().st_mtime > built
