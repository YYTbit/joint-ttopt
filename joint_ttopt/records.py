import fcntl
import hashlib
import json
import os
from pathlib import Path

from .storage import json_value, record_digest, write_json


class FileLock:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("The output is already in use") from None

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Collection:
    """Commit related trajectory records and their cursor together."""

    def __init__(self, output, identity, files):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "collection.json"
        self.handles, self.checksums = {}, {}
        self.lock = FileLock(self.output / "collection.lock")
        try:
            if self.path.is_file():
                self.state = json.loads(self.path.read_text())
                if self.state["identity"] != record_digest(identity) or set(self.state["files"]) != set(files):
                    raise ValueError("Collection resume requires the original model, inputs and settings")
            else:
                if any((self.output / name).exists() for name in files):
                    raise FileExistsError("Existing records have no committed collection cursor")
                self.state = {"identity": record_digest(identity), "cursor": 0, "complete": False,
                              "files": {name: {"offset": 0, "sha256": hashlib.sha256().hexdigest()}
                                        for name in files}}
                write_json(self.path, self.state)
            for name in files:
                target = self.output / name
                handle = target.open("r+b" if target.exists() else "w+b")
                self.handles[name] = handle
                expected = self.state["files"][name]
                remaining, checksum = expected["offset"], hashlib.sha256()
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Committed collection records are missing")
                    checksum.update(chunk)
                    remaining -= len(chunk)
                if checksum.hexdigest() != expected["sha256"]:
                    raise ValueError("Committed collection records have changed")
                handle.truncate(expected["offset"])
                self.checksums[name] = checksum
        except BaseException:
            self.close()
            raise

    @property
    def cursor(self):
        return self.state["cursor"]

    def commit(self, records, complete=False):
        if self.state["complete"]:
            raise ValueError("The collection is already complete")
        positions = {}
        for name, handle in self.handles.items():
            for record in records.get(name, []):
                body = (json.dumps(json_value(record), ensure_ascii=False, allow_nan=False) + "\n").encode()
                handle.write(body)
                self.checksums[name].update(body)
            handle.flush()
            os.fsync(handle.fileno())
            positions[name] = {"offset": handle.tell(), "sha256": self.checksums[name].hexdigest()}
        state = {**self.state, "files": positions, "cursor": self.cursor + (not complete), "complete": complete}
        write_json(self.path, state)
        self.state = state

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
