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

import spacy
nlp = spacy.load("en_core_web_sm")  # Load English model


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

def chunk_entities(entities: List[Dict[str, int]], start: int, end: int) -> List[Dict[str, int]]:
    """Filters entities that fall within the given start and end range."""
    return [e for e in entities if start <= e["entity_start"] <= e["entity_end"] <= end]

def find_sentence_token_indexes(text_indexes, offsets):
    indexes = []
    for start, end in text_indexes:
        found = False
        token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start <= start and tok_end >= start:
                found = True
            
            if not found:
                continue

            token_indices.append(i)
            if tok_end >= end:
                break

        indexes.append((token_indices[0], token_indices[-1] + 1))  # +1 for non-inclusive bounds
    return indexes

def find_entity_token_indexes(
    tokens: List[int],
    offsets: List[Tuple[int, int]],
    entities: List[Dict[str, int]],
    start_text_offset: int,
    tokenizer
) -> Tuple[List[Dict[str, int]], List[int]]:
    """Finds token indexes for entities in a chunk of text."""
    entities = deepcopy(entities)
    entity_token_indexes = []
    tokenized_text = tokenizer.decode(tokens)

    for entity in entities:
        start_text = entity["entity_start"] - start_text_offset
        end_text = entity["entity_end"] - start_text_offset
        
        start_token = next(
            (i for i, (s, e) in enumerate(offsets) if s <= start_text < e), None
        )
        end_token = next(
            (i for i, (s, e) in enumerate(offsets) if s < end_text <= e), None
        )
        
        if start_token is None or end_token is None:
            continue
        
        entity.update({
            "entity_text_start": start_text,
            "entity_text_end": end_text,
            "entity_token_start": start_token,
            "entity_token_end": end_token + 1,  # Non-inclusive end
        })
        del entity["entity_start"], entity["entity_end"]
        
        extracted_text = tokenized_text[start_text:end_text]
        decoded_tokens_text = tokenizer.decode(tokens[start_token:end_token + 1]).strip()
        
        assert extracted_text == entity["entity_text"], f"Entity text mismatch: {extracted_text} != {entity['entity_text']}"
        # if "�" not in decoded_tokens_text:
        assert extracted_text in decoded_tokens_text, f"Entity text mismatch: {extracted_text} not in {decoded_tokens_text}"
        
        entity_token_indexes.append(entity)
    
    return entity_token_indexes

def chunk_section(
    section: Dict[str, str], 
    entities: List[Dict[str, int]], 
    tokenizer, 
    chunk_size: int = 2048
) -> Tuple[List[List[Dict[str, int]]], List[List[int]]]:
    """
    Finds token indexes for entities in a section, handling large texts by chunking.
    """
    text = section["content"]
    section_start = section["begin"]
    tokens, offsets = tokenizer.encode(text, add_special_tokens=True)
    
    if len(tokens) <= chunk_size:
        entities_chunk = chunk_entities(entities, section_start, section_start + len(text))
        entity_token_indexes = find_entity_token_indexes(tokens, offsets, entities_chunk, section_start, tokenizer)
        return [entity_token_indexes], [tokens]

    # Sentence-based chunking
    doc = nlp(text)
    sentence_text_indexes = [(sent.start_char, sent.end_char) for sent in doc.sents]
    # Because the tokenizer removes beginning whitespace, we need to update the offset of the first sentence.
    if offsets[0][0] > 0:
        sentence_text_indexes[0] = (offsets[0][0], sentence_text_indexes[0][1])

    sentence_token_indexes = find_sentence_token_indexes(sentence_text_indexes, offsets)

    chunk_text_indexes = []
    chunk_start, chunk_end = -1, -1

    # Find the token bounds of each chunk.
    for (start_text, end_text), (start_token, end_token) in zip(sentence_text_indexes, sentence_token_indexes):
        if chunk_start == -1:
            chunk_start, chunk_end = start_text, end_text
            chunk_token_start = start_token
        
        if end_token - chunk_token_start > chunk_size:
            chunk_text_indexes.append((chunk_start, chunk_end))
            chunk_start = start_text
            chunk_token_start = start_token
        chunk_end = end_text
    
    chunk_text_indexes.append((chunk_start, chunk_end))

    entity_token_indexes, all_chunk_tokens = [], []

    for chunk_start, chunk_end in chunk_text_indexes:
        # We must recompute the chunk offsets because the tokenizer removes repeated whitespace.
        chunk_tokens, chunk_offsets = tokenizer.encode(text[chunk_start:chunk_end], add_special_tokens=True)
        
        entities_chunk = chunk_entities(entities, section_start + chunk_start, section_start + chunk_end)
        entity_token_indexes_chunk = find_entity_token_indexes(chunk_tokens, chunk_offsets, entities_chunk, section_start + chunk_start, tokenizer)
        
        if len(chunk_tokens) > chunk_size:
            # This can happen for long sentences.
            # We choose to drop here, we can decide to do something smarter later...
            # There are < 3K affected chunks out of > 6M chunks.
            continue

        entity_token_indexes.append(entity_token_indexes_chunk)
        all_chunk_tokens.append(chunk_tokens)
    
    return entity_token_indexes, all_chunk_tokens

def chunk_document(
    text: str,
    entities: List[Dict[str, int]], 
    tokenizer, 
    chunk_size: int = 2048
):
    section = {"content": text, "begin": 0}
    return chunk_section(section, entities, tokenizer, chunk_size)

def tokenize_file(
        tokenizer_name_or_path: str,
        path: str,
        refresh_tokenizer_every: int = 0,
        chunk_size: int = 2048,
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
                    if row.namespace != ARTICLE_NAMESPACE:
                        continue
                    if row.text.strip():
                        # skip empty docs
                        if tokenizer.mode == "sections":
                            for section in row.sections:
                                entities, chunks = chunk_section(section, row.entities, tokenizer, chunk_size)
                        elif tokenizer.mode == "vsl":
                            entities, chunks = chunk_document(row.text, row.entities, tokenizer, sys.maxsize)
                        else:
                            entities, chunks = chunk_document(row.text, row.entities, tokenizer, chunk_size)

                        for chunk_entities, chunk_tokens in zip(entities, chunks):
                            if refresh_tokenizer_every:
                                # extra copy to prevent memory leaks
                                chunk_tokens = np.array(chunk_tokens, dtype=dtype)
                            yield KASTokenizerOutput.from_tokens(id=row.id, src=path, loc=loc, tokens=chunk_tokens, title=row.title, entities=chunk_entities) # pyright: ignore
                            loc += 1

                    if refresh_tokenizer_every > 0 and loc % refresh_tokenizer_every == 0:
                        # to prevent memory leaks, we refresh the tokenizer every so often
                        del tokenizer
                        gc.collect()
                        tokenizer = make_tokenizer(tokenizer_name_or_path, **tokenizer_kwargs)

                except Exception as ex:
                    logger.error("Error processing %s:%d", path, row.id, exc_info=ex)
    