from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends

from ..storage import Storage, open_storage
from .security import security

__all__ = ["get_storage"]


_storage: Optional[Storage] = None

DEFAULT_SECURITY = Depends(security)


def get_storage() -> Storage:
    global _storage  # noqa: PLW0603e
    if _storage is None:
        return (_storage := open_storage(os.environ["ERT_STORAGE_ENS_PATH"]))
    _storage.refresh()
    return _storage
