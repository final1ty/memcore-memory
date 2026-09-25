"""Backwards-compatible alias for the pre-rename ``mnemosyne`` package.

Re-exporting a handful of top-level names is not enough: released code and the
README both do ``from mnemosyne.embeddings.factory import get_embedder``, which
needs the whole submodule tree. A meta-path finder maps every ``mnemosyne.X`` to
the already-imported ``memcore_memory.X`` object, so the two names share one module
instance and ``isinstance`` keeps working across them.
"""

import importlib
import importlib.util
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
        # find_spec, not import: an ImportError raised inside a real module must
        # surface as itself, not as "No module named mnemosyne.X".
        real = importlib.util.find_spec(_TARGET + fullname[len(_ALIAS):])
        if real is None:
            return None
        # The origin is what runpy turns into __main__.__file__ under `-m`; the
        # shared module object keeps its own __file__ either way.
        return ModuleSpec(fullname, self, origin=real.origin)

    def create_module(self, spec):
        module = importlib.import_module(_TARGET + spec.name[len(_ALIAS):])
        # module_from_spec() overwrites __spec__ with this alias spec whatever
        # create_module returns; keep the real one to put back in exec_module.
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        # Already executed under its real name. The alias spec left in __spec__
        # made importlib.reload() of the real module a silent no-op.
        real = getattr(module.__spec__, "loader_state", None)
        if real is not None:
            module.__spec__ = real

    # runpy asks the loader for code rather than a module, so without these
    # `python -m mnemosyne.cli.main` failed once this finder came first.
    @staticmethod
    def _real_loader(fullname):
        real = _TARGET + fullname[len(_ALIAS):]
        return real, importlib.util.find_spec(real).loader

    def get_code(self, fullname):
        real, loader = self._real_loader(fullname)
        return loader.get_code(real)

    def get_source(self, fullname):
        real, loader = self._real_loader(fullname)
        return loader.get_source(real)

    def is_package(self, fullname):
        real, loader = self._real_loader(fullname)
        return loader.is_package(real)


# First, not last: after PathFinder, `mnemosyne.X.Y` was found through the shared
# parent's __path__ and executed a second time, giving duplicate classes
# (isinstance and except failed across the two names) and replacing attributes
# of the real package with the copies.
if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

__all__ = ["MnemosyneMemory", "MemcoreMemory", "create_memory_system", "__version__"]
