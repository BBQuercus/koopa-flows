from os.path import exists

import koopa.io
import pandas as pd
import xxhash
from cpr.Resource import Resource
from cpr.Target import Target
from pandas.core.util.hashing import hash_pandas_object


class ParquetSource(Resource):

    def __init__(
            self,
            location: str,
            name: str,
            ext: str
    ):
        super(ParquetSource, self).__init__(
            location=location,
            name=name,
            ext=ext
        )

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            assert exists(self.get_path()), f"{self.get_path()} does not " \
                                            f"exist."

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
            location=location,
            name=name,
            ext=ext,
            data_hash=data_hash
        )

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            assert exists(self.get_path()), f"{self.get_path()} does not " \
                                            f"exist."

            self._data = koopa.io.load_parquet(self.get_path())

        return self._data

    def _hash_data(self, data) -> str:
        return xxhash.xxh3_64(
            hash_pandas_object(data).values.tobytes()).hexdigest()

    def _write_data(self):
        if self._data is not None:
            koopa.io.save_parquet(self.get_path(), self._data)
