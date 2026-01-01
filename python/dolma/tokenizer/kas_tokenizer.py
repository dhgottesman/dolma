from .tokenizer import (
    Tokenizer,
    logger
)
import sys
import os
import gc
from .data_types import KASInputSpec, KASTokenizerOutput
from typing import TYPE_CHECKING, Generator, List, Tuple, Dict
from copy import deepcopy
import msgspec
import numpy as np
import smart_open


ARTICLE_NAMESPACE = "Article"


class KASTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop("mode", "")
        super().__init__(*args, **kwargs)

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        encoding = self.base_tokenizer.encode(input, add_special_tokens=False)
        input_ids, offsets = encoding.ids, encoding.offsets
        
        if add_special_tokens:
            input_ids = self.add_special_tokens(input_ids)
        return input_ids, offsets

def make_tokenizer(
    tokenizer_name_or_path: str,
    **tokenizer_kwargs,
) -> KASTokenizer:
    tokenizer = (
        KASTokenizer.from_file(tokenizer_name_or_path, **tokenizer_kwargs)
        if os.path.exists(tokenizer_name_or_path) and os.path.isfile(tokenizer_name_or_path)
        else KASTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    )
    return tokenizer

def add_token_indices(
    tokens: List[int],
    offsets: List[Tuple[int, int]],
    entities: List[Dict[str, int]],
    tokenizer
) -> Tuple[List[Dict[str, int]], List[int]]:
    """Finds token indexes for entities in a chunk of text."""
    entities = deepcopy(entities)
    entity_token_indexes = []
    tokenized_text = tokenizer.decode(tokens)

    for entity in entities:
        start_text = entity["char_start"]
        end_text = entity["char_end"]
        
        start_token = next(
            (i for i, (s, e) in enumerate(offsets) if s <= start_text < e), None
        )
        end_token = next(
            (i for i, (s, e) in enumerate(offsets) if s < end_text <= e), None
        )
        
        if start_token is None or end_token is None:
            continue
        
        entity.update({
            "tok_start": start_token,
            "tok_end": end_token + 1,  # Non-inclusive end
        })
        
        extracted_text = tokenized_text[start_text:end_text]
        decoded_tokens_text = tokenizer.decode(tokens[start_token:end_token + 1])
        
        assert extracted_text == entity["text"], f"Entity text mismatch: {extracted_text} != {entity['text']}"
        # if "�" not in decoded_tokens_text:
        assert extracted_text.strip() in decoded_tokens_text.strip(), f"Entity text mismatch: {extracted_text.strip()} not in {decoded_tokens_text.strip()}"
        
        entity_token_indexes.append(entity)
    
    return entity_token_indexes

def tokenize_file(
        tokenizer_name_or_path: str,
        path: str,
        refresh_tokenizer_every: int = 0,
        **tokenizer_kwargs,
    ) -> Generator[KASTokenizerOutput, None, None]:
        """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
        file, each containing a field named `text`.
        """
        tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)
        dtype = deepcopy(tokenizer.dtype)
        decoder = msgspec.json.Decoder(KASInputSpec)
        loc = 0
        with smart_open.open(path, mode="rt") as input_stream:
            for _, line in enumerate(input_stream, start=1):
                try:
                    row = decoder.decode(line)

                    if text := row.text.strip(): # Skip empty docs
                        entities = row["entities"]
                        tokens, offsets = tokenizer.encode(text, add_special_tokens=True) # Daniela, should change to False
                        entities = add_token_indices(tokens, offsets, entities, tokenizer)
                        
                        if refresh_tokenizer_every:
                            # extra copy to prevent memory leaks
                            tokens = np.array(tokens, dtype=dtype)
                            yield KASTokenizerOutput.from_tokens(id=row.id, src=path, loc=loc, tokens=tokens, title=row.title, entities=entities, offsets=offsets) # pyright: ignore
                            loc += 1

                    if refresh_tokenizer_every > 0 and loc % refresh_tokenizer_every == 0:
                        # to prevent memory leaks, we refresh the tokenizer every so often
                        del tokenizer
                        gc.collect()
                        tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)

                except Exception as ex:
                    logger.error("Error processing %s:%s", path, row.id, exc_info=ex)
    