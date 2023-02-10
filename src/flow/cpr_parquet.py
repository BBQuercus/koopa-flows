import os

from cpr.Resource import Resource
from cpr.Target import Target
import koopa.io
import pandas as pd
import xxhash


class ParquetSource(Resource):
    def __init__(self, location: str, name: str, ext: str):
        super(ParquetSource, self).__init__(location=location, name=name, ext=ext)

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            assert os.path.exists(self.get_path()), f"{self.get_path()} does not exist."
            self._data = koopa.io.load_parquet(self.get_path())

        return self._data


class ParquetTarget(Target):
    def __init__(
        self,
        location: str,
        name: str,
        ext: str = ".parq",
        data_hash: str = None,
    ):
        super(ParquetTarget, self).__init__(
            location=location, name=name, ext=ext, data_hash=data_hash
        )

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            assert os.path.exists(self.get_path()), (
                f"{self.get_path()} does not " f"exist."
            )
            self._data = koopa.io.load_parquet(self.get_path())

        return self._data

    def _hash_data(self, data) -> str:
        data_hash = pd.core.util.hashing.hash_pandas_object(data).values.tobytes()
        return xxhash.xxh3_64(data_hash).hexdigest()

    def _write_data(self):
        if self._data is not None:
            koopa.io.save_parquet(self.get_path(), self._data)
