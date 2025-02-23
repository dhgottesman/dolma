import unittest
import json
from dolma.tokenizer.tokenizer import (
    find_entity_token_indexes,
    make_tokenizer,
    clean_sections,
    chunk_entities
)

class TestTokenizerFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load test data from JSON file once for all test cases."""
        with open("dolma/tests/data.json", "r") as file:
            cls.test_data = json.load(file)
        
        cls.tokenizer = make_tokenizer("EleutherAI/gpt-neox-20b")
        cls.sections, cls.entities = [], []
        for test_case in cls.test_data:
            cls.sections.append(clean_sections(test_case["sections"]))
            cls.entities.append(test_case["entities"])

    # def test_find_entity_token_indexes(self):
    #     for i, (all_entities, sections) in enumerate(zip(self.entities, self.sections)):
    #         for j, section in enumerate(sections):
    #             section_entities = chunk_entities(all_entities, section["begin"], section["end"])
    #             section_entities, chunks = find_entity_token_indexes(section, section_entities, self.tokenizer)
                
    #             for entities, chunk_tokens in zip(section_entities, chunks):
    #                 for entity in entities:
    #                     entity_tokens = chunk_tokens[entity["entity_token_start"]:entity["entity_token_end"]]
    #                     self.assertIn(entity["entity_text"].strip(), self.tokenizer.decode(entity_tokens).strip())

    def test_single(self):
            sections = self.sections[2]
            all_entities = self.entities[2]
            for section in sections:
                section_entities = chunk_entities(all_entities, section["begin"], section["end"])
                section_entities, chunks = find_entity_token_indexes(section, section_entities, self.tokenizer)
                
                for entities, chunk_tokens in zip(section_entities, chunks):
                    for entity in entities:
                        entity_tokens = chunk_tokens[entity["entity_token_start"]:entity["entity_token_end"]]
                        self.assertIn(entity["entity_text"].strip(), self.tokenizer.decode(entity_tokens).strip())


if __name__ == "__main__":
    unittest.main()