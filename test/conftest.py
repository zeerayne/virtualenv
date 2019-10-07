from __future__ import absolute_import, unicode_literals

import os
import sys

import coverage
import pytest
from pathlib2 import Path

from virtualenv.util import symlink


@pytest.fixture(autouse=True)
def check_cwd_not_changed_by_test():
    old = os.getcwd()
    yield
    new = os.getcwd()
    if old != new:
        pytest.fail("test changed cwd: {!r} => {!r}".format(old, new))


@pytest.fixture(autouse=True)
def check_os_environ_stable():
    old = os.environ.copy()
    # ensure we don't inherit parent env variables
    to_clean = {
        k for k in os.environ.keys() if k.startswith("VIRTUALENV_") or "VIRTUAL_ENV" in k or k.startswith("TOX_")
    }
    cleaned = {k: os.environ[k] for k, v in os.environ.items()}
    os.environ[str("VIRTUALENV_NO_DOWNLOAD")] = str("1")
    is_exception = False
    try:
        yield
    except BaseException:
        is_exception = True
        raise
    finally:
        try:
            del os.environ[str("VIRTUALENV_NO_DOWNLOAD")]
            if is_exception is False:
                new = os.environ
                extra = {k: new[k] for k in set(new) - set(old)}
                miss = {k: old[k] for k in set(old) - set(new) - to_clean}
                diff = {
                    "{} = {} vs {}".format(k, old[k], new[k])
                    for k in set(old) & set(new)
                    if old[k] != new[k] and not k.startswith("PYTEST_")
                }
                if extra or miss or diff:
                    msg = "test changed environ"
                    if extra:
                        msg += " extra {}".format(extra)
                    if miss:
                        msg += " miss {}".format(miss)
                    if diff:
                        msg += " diff {}".format(diff)
                    pytest.fail(msg)
        finally:
            os.environ.update(cleaned)


COV_ENV_VAR = "COVERAGE_PROCESS_START"
COVERAGE_RUN = os.environ.get(COV_ENV_VAR)
COV_FOLDERS = (
    [i for i in Path(coverage.__file__).parents[1].iterdir() if i.name.startswith("coverage")] if COVERAGE_RUN else None
)


@pytest.fixture(autouse=True)
def enable_coverage_in_virtual_env(monkeypatch):
    """
    Enable coverage report collection on the created virtual environments by injecting the coverage project
    """
    if COVERAGE_RUN:
        from virtualenv import run

        _original_run_create = run._run_create

        def _our_run(creator):
            _original_run_create(creator)
            enable_coverage_on_env(monkeypatch, creator)  # now inject coverage tools

        try:
            run._run_create = _our_run
            yield
        finally:
            run._run_create = _original_run_create
    else:
        yield


def enable_coverage_on_env(monkeypatch, creator):
    site_packages = creator.site_packages[0]
    for folder in COV_FOLDERS:
        target = site_packages / folder.name
        if not target.exists():
            symlink(folder, target)
    if sys.version_info[0] == 2:
        # coverage for the injected site.py on Python 2
        monkeypatch.setenv(str("_VIRTUALENV_INJECTED_SRC"), str(site_packages.parent / "site.py"))
    (site_packages / "coverage-virtualenv.pth").write_text("import coverage; coverage.process_startup()")
