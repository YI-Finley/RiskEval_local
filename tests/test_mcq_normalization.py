from __future__ import annotations

import unittest

from riskeval.parsing import normalize_answer


class MCQNormalizationTests(unittest.TestCase):
    def test_normalize_plain_choice_letter(self) -> None:
        self.assertEqual(normalize_answer("D", "mcq"), "D")

    def test_normalize_labeled_choice_letter(self) -> None:
        self.assertEqual(normalize_answer("answer: d (~33.4)", "mcq"), "D")

    def test_normalize_choice_prefix(self) -> None:
        self.assertEqual(normalize_answer("D. mitochondria", "mcq"), "D")


if __name__ == "__main__":
    unittest.main()
