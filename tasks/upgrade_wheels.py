"""
Helper script to rebuild virtualenv_support. Downloads the wheel files using pip
"""
from __future__ import absolute_import, unicode_literals

import os
import subprocess
import sys
from collections import defaultdict

from pathlib2 import Path
from tempfile import TemporaryDirectory
from threading import Thread
import shutil

STRICT = "UPGRADE_ADVISORY" not in os.environ

BUNDLED = ["pip", "setuptools", "wheel"]
SUPPORT = [(2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]
DEST = Path(__file__).resolve().parents[1] / "src" / "virtualenv" / "wheels"


def download(ver, dest, package):
    subprocess.call(
        [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", ver, "-d", dest, package]
    )


def run():
    old_batch = set(i.name for i in DEST.iterdir() if i.name != "__init__.py")
    with TemporaryDirectory() as temp:
        temp_path = Path(temp)
        folders = {}
        targets = []
        for support in SUPPORT:
            support_ver = ".".join(str(i) for i in support)
            into = temp_path / support_ver
            into.mkdir()
            folders[into] = support
            for package in BUNDLED:
                thread = Thread(target=download, args=(support_ver, str(into), package))
                targets.append(thread)
                thread.start()
        for thread in targets:
            thread.join()
        new_batch = {i.name: i for f in folders.keys() for i in Path(f).iterdir()}

        new_packages = new_batch.keys() - old_batch
        remove_packages = old_batch - new_batch.keys()

        for package in remove_packages:
            (DEST / package).unlink()
        for package in new_packages:
            shutil.copy2(str(new_batch[package]), DEST / package)

        added = collect_package_versions(new_packages)
        removed = collect_package_versions(remove_packages)

        outcome = (1 if STRICT else 0) if (added or removed) else 0
        for key, versions in added.items():
            text = "* upgrade embedded {} to {}".format(key, fmt_version(versions))
            if key in removed:
                text += " from {}".format(removed[key])
                del removed[key]
            print(text)
        for key, versions in removed.items():
            print("* removed embedded {} of {}".format(key, fmt_version(versions)))

        support_table = defaultdict(list)
        for package in new_batch.keys():
            for folder, version in folders.items():
                if (folder / package).exists():
                    support_table[version].append(package)
        support_table = {k: {i.split("-")[0]: i for i in v} for k, v in support_table.items()}
        dest_target = DEST / "__init__.py"
        dest_target.write_text("SUPPORT = {!r}; MAX = {!r}".format(support_table, max(sorted(SUPPORT))))
        subprocess.check_call([sys.executable, "-m", "black", str(dest_target)])

        raise SystemExit(outcome)


def fmt_version(versions):
    return ", ".join("``{}``".format(v) for v in versions)


def collect_package_versions(new_packages):
    result = defaultdict(list)
    for package in new_packages:
        split = package.split("-")
        if len(split) < 2:
            raise ValueError(package)
        key, version = split[0:2]
        result[key].append(version)
    return result


if __name__ == "__main__":
    run()
