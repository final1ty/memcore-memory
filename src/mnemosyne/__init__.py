"""Backwards-compatible alias for the pre-rename ``mnemosyne`` package.

Re-exporting a handful of top-level names is not enough: released code and the
README both do ``from mnemosyne.embeddings.factory import get_embedder``, which
needs the whole submodule tree. A meta-path finder maps every ``mnemosyne.X`` to
the already-imported ``memcore_memory.X`` object, so the two names share one module
instance and ``isinstance`` keeps working across them.
"""

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

from memcore_memory import __version__
from memcore_memory.core.memory import MnemosyneMemory as MnemosyneMemory
from memcore_memory.core.memory import MnemosyneMemory as MemcoreMemory

try:
    from memcore_memory import create_memory_system
except ImportError:  # pragma: no cover - optional backends may be absent
    pass

_ALIAS = "mnemosyne."
_TARGET = "memcore_memory."


class _AliasFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_ALIAS):
            return None
        return ModuleSpec(fullname, self)

    def create_module(self, spec):
        module = importlib.import_module(_TARGET + spec.name[len(_ALIAS):])
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass  # already executed under its real name


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_AliasFinder())

__all__ = ["MnemosyneMemory", "MemcoreMemory", "create_memory_system", "__version__"]
