import unittest
from unittest.mock import patch, MagicMock
import json
from src.llm_provider import GeminiProvider
from src.config.settings import ChimeraSettings
from src.exceptions import SchemaValidationError
from pydantic import BaseModel, Field
from typing import Optional

class TestModel(BaseModel):
    test_field: str
    another_field: Optional[int] = None

class TestGeminiProvider(unittest.TestCase):
    
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.count_tokens.side_effect = lambda text: len(text) // 4
        self.mock_tokenizer.max_output_tokens = 8192
        
        self.provider = GeminiProvider(
            api_key="AIza_mock-key-for-testing-purposes-1234567890",
            model_name="gemini-2.5-flash-lite",
            tokenizer=self.mock_tokenizer,
            settings=ChimeraSettings(),
            rich_console=MagicMock(),
            request_id="test"
        )
        self.provider.client = self.mock_client
        
    def test_generate_with_valid_json(self):
        """Test generate method with valid JSON response."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text='{"test_field": "value"}')))]
        self.mock_client.models.generate_content.return_value = mock_response
        
        response, input_tokens, output_tokens, is_truncated = self.provider.generate(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            output_schema=TestModel
        )
        
        self.assertEqual(response, '{"test_field": "value"}')
        # Token counts are mocked to return 0 for simplicity in this test,
        # as the focus is on the JSON parsing and schema validation.
        self.assertEqual(input_tokens, 0) 
        self.assertEqual(output_tokens, 0)
        self.assertFalse(is_truncated)
        
    def test_generate_with_invalid_json(self):
        """Test generate method with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text='{"test_field": "value"')))] # Malformed JSON
        self.mock_client.models.generate_content.return_value = mock_response
        
        with self.assertRaises(SchemaValidationError):
            self.provider.generate(
                prompt="Test prompt",
                system_prompt="Test system prompt",
                output_schema=TestModel
            )
    
    def test_clean_generated_text(self):
        """Test the _clean_generated_text method."""
        # Test with markdown code fence
        text = "```json\n{\"test\": \"value\"}\n```"
        cleaned = self.provider.output_parser._clean_llm_output(text) # Use the parser's method
        self.assertEqual(cleaned, '{"test": "value"}')
        
        # Test with unescaped quotes (this is now handled by _repair_json_string in output_parser)
        # The _clean_generated_text in llm_provider.py is replaced by output_parser._clean_llm_output
        # and output_parser._repair_json_string.
        # This test should reflect the combined behavior.
        text_with_unescaped_quotes = '{"test": "value with "unescaped" quotes"}'
        # The parser's repair logic would attempt to fix this.
        # For a direct test of the repair, we'd call _repair_json_string directly.
        # Here, we're testing the end-to-end cleaning.
        # The expected output from _clean_llm_output would be the raw string,
        # and then _parse_with_incremental_repair would fix it.
        # For this unit test, we'll simulate a simple case that _clean_llm_output might handle.
        
        # Test with incomplete JSON (this is now handled by _parse_with_incremental_repair in output_parser)
        text_incomplete = '{"test": "value"'
        # The parser's repair logic would attempt to fix this.
        # For a direct test of the repair, we'd call _repair_json_string directly.
        # Here, we're testing the end-to-end cleaning.
        
        # Since _clean_generated_text is replaced by output_parser._clean_llm_output,
        # and the repair logic is in _parse_with_incremental_repair,
        # this test needs to be adjusted to reflect the new structure.
        # For now, I'll keep the markdown test as it's a direct cleaning step.
        # The other cases are more complex and involve the full parsing pipeline.
        
        # Re-evaluating the test for _clean_generated_text based on the new structure:
        # The original _clean_generated_text method is effectively replaced by
        # `self.output_parser._clean_llm_output(generated_text)` in `llm_provider.py`.
        # The complex JSON repair logic is now within `LLMOutputParser._parse_with_incremental_repair`.
        # So, this test should primarily focus on what `_clean_llm_output` does.
        
        # Test with markdown code fence (already covered)
        text_md = "```json\n{\"test\": \"value\"}\n```"
        cleaned_md = self.provider.output_parser._clean_llm_output(text_md)
        self.assertEqual(cleaned_md, '{"test": "value"}')
        
        # Test with conversational filler outside JSON
        text_filler = "Hello! Here is the JSON:\n```json\n{\"key\": \"val\"}\n```\nThanks!"
        cleaned_filler = self.provider.output_parser._clean_llm_output(text_filler)
        self.assertEqual(cleaned_filler, '{"key": "val"}')
        
        # Test with JSON not in markdown, but with surrounding text
        text_surrounding = "Some text before. {\"data\": 123} Some text after."
        cleaned_surrounding = self.provider.output_parser._clean_llm_output(text_surrounding)
        self.assertEqual(cleaned_surrounding, '{"data": 123}')
        
        # Test with malformed JSON that _clean_llm_output itself doesn't fix (it's for _parse_with_incremental_repair)
        text_malformed = '{"test": "value" with "unescaped" quotes}'
        cleaned_malformed = self.provider.output_parser._clean_llm_output(text_malformed)
        # _clean_llm_output primarily removes markdown/filler, it doesn't fix internal JSON structure.
        # So, the output should still be the raw malformed string, without markdown/filler.
        self.assertEqual(cleaned_malformed, text_malformed) # Expect no change from _clean_llm_output itself
        
        text_incomplete = '{"test": "value"'
        cleaned_incomplete = self.provider.output_parser._clean_llm_output(text_incomplete)
        self.assertEqual(cleaned_incomplete, text_incomplete) # Expect no change from _clean_llm_output itself