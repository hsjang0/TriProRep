import json
import os
import pickle
from typing import Any, Iterable, Tuple

import lmdb


def decode_lmdb_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return pickle.loads(value)
        except Exception:
            try:
                return json.loads(value.decode("utf-8"))
            except Exception:
                return value
    return value


def write_lmdb(
    results: Iterable[Tuple[str, dict]],
    lmdb_path: str,
    metadata: dict,
    map_size: int = 1099511627776 * 2,
) -> None:
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for protein_id, result in results:
            txn.put(protein_id.encode(), pickle.dumps(result))
        txn.put(b"__metadata__", pickle.dumps(metadata))
    env.close()
