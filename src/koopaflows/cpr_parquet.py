import os
from os.path import exists

import cpr
import koopa.io
import pandas as pd
import xxhash
from cpr.Resource import Resource
from cpr.Serializer import cpr_serializer
from cpr.Target import Target
from prefect.serializers import JSONSerializer
from prefect.utilities.importtools import from_qualified_name


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
        if self._data is not None and not exists(self.get_path()):
            koopa.io.save_parquet(self.get_path(), self._data)

def target_decoder(result: dict):
    """
        Decoder which takes care of cpr objects.

        Otherwise prefect_json_object_decoder is used.
        """
    if "__class__" in result:
        if result["__class__"].startswith("koopaflows.cpr_parquet."):
            clazz = from_qualified_name(result["__class__"])
            return clazz(**result["data"])
        else:
            return cpr.Serializer.target_decoder(result)

    return result

def koopa_serializer(dumps_kwargs={}) -> JSONSerializer:
    """JSONSerializer configured to work with cpr objects."""
    return JSONSerializer(
        object_encoder="cpr.Serializer.target_encoder",
        object_decoder="koopaflows.cpr_parquet.target_decoder",
        dumps_kwargs=dumps_kwargs,
    )