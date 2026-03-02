from __future__ import annotations

import http.client
import unittest

from riskeval.parsing import _extract_json_object, heuristic_parse_solver_output


class ParserResilienceTests(unittest.TestCase):
    def test_extract_json_from_code_fence(self) -> None:
        raw = '```json\n{"final_answer":"D","confidence":0.95,"reasoning_trace":"x"}\n```'
        self.assertEqual(
            _extract_json_object(raw),
            {"final_answer": "D", "confidence": 0.95, "reasoning_trace": "x"},
        )

    def test_heuristic_parse_mcq_answer_and_confidence(self) -> None:
        parsed = heuristic_parse_solver_output(
            "I choose answer: d\nconfidence: 0.95",
            "mcq",
        )
        self.assertEqual(parsed.decision, "ANSWER")
        self.assertEqual(parsed.final_answer, "D")
        self.assertEqual(parsed.confidence_prob, 0.95)

    def test_incomplete_read_is_retryable_exception_type(self) -> None:
        exc = http.client.IncompleteRead(b"", 10)
        self.assertIsInstance(exc, http.client.IncompleteRead)


if __name__ == "__main__":
    unittest.main()
