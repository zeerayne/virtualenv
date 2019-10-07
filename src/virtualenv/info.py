from __future__ import absolute_import, unicode_literals

import os
import sys

IS_PYPY = hasattr(sys, "pypy_version_info")
PY3 = sys.version_info[0] == 3
IS_WIN = sys.platform == "win32"


USER_DIR = os.path.expanduser("~")
if IS_WIN:
    DEFAULT_STORAGE_DIR = os.path.join(USER_DIR, "virtualenv")
else:
    DEFAULT_STORAGE_DIR = os.path.join(USER_DIR, ".virtualenv")
