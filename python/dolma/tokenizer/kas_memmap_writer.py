import functools
import os
import re
from contextlib import ExitStack
from csv import writer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, TextIO

import numpy as np
import smart_open

from ..core.loggers import get_logger
from .data_types import KASMemmapMetadata, KASTokenizerOutput
from .memmap_writer import MemmapWriter

log = get_logger(__name__)


class KASMemmapWriter(MemmapWriter):

    def __init__(self, *args, **kwargs):
        super(KASMemmapWriter, self).__init__(*args, **kwargs)

    def write(self, output: KASTokenizerOutput, flush: bool = True) -> bool:
        """Write a list of token IDs to the memmap file; if only a subset of the values can be written,
        return the rest.

        Args:
            values (List[int]): List of token IDs to write.
            flush (bool, optional): Whether to flush the memmap file after writing. Defaults to False.
        """

        if self._memmap_file is None:
            raise RuntimeError("MemmapFile is not open")

        if self._metadata_file is None:
            raise RuntimeError("Metadata file is not open")

        if (len(output.tokens) + self._written_tokens) >= self.max_tokens:
            # return false if the memmap file is full
            return False

        metadata = KASMemmapMetadata(
            id=output.id,
            src=output.src,
            loc=output.loc,
            start=self._written_tokens,
            end=self._written_tokens + output.end,
            title=output.title,
            entities=output.entities,
            offsets=output.offsets,
        )
        self._memmap_file[self._written_tokens : self._written_tokens + output.end] = output.tokens
        self._written_tokens += output.end

        # self._metadata_file.write(msgspec.json.encode(metadata) + b"\n")
        self.metadata_writer.writerow(metadata)

        if flush:
            self.flush()

        return True