import json
import csv
from io import StringIO
from typing import List, NamedTuple, Dict

from ..core.data_types import KASInputSpec

__all__ = ["KASInputSpec", "KASTokenizerOutput", "KASMetadata"]

class KASTokenizerOutput(NamedTuple):
    id: int
    src: str
    loc: int
    title: str
    tokens: List[int]
    entities: List[Dict]
    start: int
    end: int

    @classmethod
    def from_tokens(cls, id: str, src: str, loc: int, tokens: List[int], title: str, entities: List[Dict]) -> "KASTokenizerOutput":
        return cls(id=id, src=src, loc=loc, tokens=tokens, start=0, end=len(tokens), title=title, entities=entities)

    @classmethod
    def from_output_spec(cls, output_spec: "KASTokenizerOutput", start: int = -1, end: int = -1) -> "KASTokenizerOutput":
        start = start if start >= 0 else output_spec.start
        end = end if end >= 0 else output_spec.end
        return cls(
            id=output_spec.id,
            src=output_spec.src,
            loc=output_spec.loc,
            tokens=output_spec.tokens,
            start=start,
            end=end,
            title=output_spec.title,
            entities=output_spec.entities,
        )

class KASMetadata(NamedTuple):
    id: str
    src: str
    loc: int
    start: int
    end: int
    title: str
    entities: List[Dict]

    def to_csv(self) -> str:
        output = StringIO()
        writer = csv.writer(output)
        entities_str = json.dumps(self.entities)
        writer.writerow([self.id, self.src, self.loc, self.start, self.end, self.title, entities_str])
        return output.getvalue().strip()

class KASMemmapMetadata(NamedTuple):
    start: int
    end: int
    id: str
    src: str
    loc: int
    title: str
    entities: List[Dict]

class TokenizerOutput(NamedTuple):
    id: str
    src: str
    loc: int
    tokens: List[int]
    start: int
    end: int

    @classmethod
    def from_tokens(cls, id: str, src: str, loc: int, tokens: List[int]) -> "TokenizerOutput":
        return cls(id=id, src=src, loc=loc, tokens=tokens, start=0, end=len(tokens))

    @classmethod
    def from_output_spec(cls, output_spec: "TokenizerOutput", start: int = -1, end: int = -1) -> "TokenizerOutput":
        start = start if start >= 0 else output_spec.start
        end = end if end >= 0 else output_spec.end
        return cls(
            id=output_spec.id,
            src=output_spec.src,
            loc=output_spec.loc,
            tokens=output_spec.tokens,
            start=start,
            end=end,
        )


class Metadata(NamedTuple):
    id: str
    src: str
    loc: int
    start: int
    end: int

    def to_csv(self) -> str:
        return f"{self.id},{self.src},{self.loc},{self.start},{self.end}"


class MemmapMetadata(NamedTuple):
    start: int
    end: int
    id: str
    src: str
    loc: int
