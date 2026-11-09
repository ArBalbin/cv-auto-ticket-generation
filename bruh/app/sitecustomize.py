import os
import sys


# VS Code debugpy can hang/crash on Python 3.12 sys.monitoring while importing
# heavy libraries such as FastAPI/Pydantic. These env vars are read by pydevd
# during debugger startup, before app code imports.
os.environ.setdefault("PYDEVD_USE_SYS_MONITORING", "0")
os.environ.setdefault("PYDEVD_USE_CYTHON", "NO")
os.environ.setdefault("PYDEVD_USE_FRAME_EVAL", "NO")
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
