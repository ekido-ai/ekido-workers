"""Ticket #141: the verifier-side check of the rows a teacher generated. Pure Python: no model, no GPU, no network."""

import json
import unittest

from ekido_verifier import generated_rows as g

TICKET = "ticket-sorter-hinglish"
SENTIMENT = "review-sentiment"
INVOICE = "invoice-fields"
FAQ = "faq-bot-hinglish"


def jsonl(rows):
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def ticket_rows(n=12):
    labels = ("complaint", "order", "question", "refund", "other")
    return [{"text": f"mera parcel number {i} abhi tak nahi aaya, kab tak milega", "label": labels[i % 5]} for i in range(n)]


def invoice_row(i=1):
    values = {"vendor": f"Shree Traders {i}", "invoice_no": f"INV-{i:04d}", "date": "2026-11-0%d" % (i % 9 + 1), "total": f"{1000 + i}.00", "gstin": "27ABCDE1234F1Z5"}
    text = f"Invoice from {values['vendor']}, no {values['invoice_no']} dated {values['date']}, GSTIN {values['gstin']}, amount payable {values['total']}"
    return {"text": text, "json": json.dumps(values)}


class CleanFiles(unittest.TestCase):
    def test_a_clean_file_passes_with_no_reasons_and_a_stable_hash(self):
        data = jsonl(ticket_rows())
        first = g.check_generated_jsonl(data, TICKET)
        self.assertEqual((first.passed, first.rows, first.violations, first.reasons), (True, 12, 0, []))
        self.assertEqual(first.row_hash, g.check_generated_jsonl(data, TICKET).row_hash)  # deterministic
        self.assertEqual(len(first.row_hash), 64)
        self.assertEqual(g.check_generated_jsonl(jsonl(ticket_rows(11)), TICKET).row_hash == first.row_hash, False)

    def test_clean_invoice_and_faq_and_other_recipes_pass(self):
        self.assertTrue(g.check_rows([invoice_row(i) for i in range(5)], INVOICE).passed)
        faq = [{"question": f"aapka clinic kab khulta hai {i}?", "answer": "Subah 9 baje se shaam 6 baje tak."} for i in range(4)]
        self.assertTrue(g.check_rows(faq, FAQ).passed)
        product = [{"attributes": f"blue kurta, cotton, size M {i}", "description": "A soft cotton kurta."} for i in range(3)]
        self.assertTrue(g.check_rows(product, "product-descriptions").passed)


class PoisonedRows(unittest.TestCase):
    def reasons(self, rows, recipe_id=TICKET, **kw):
        verdict = g.check_rows(rows, recipe_id, **kw)
        self.assertFalse(verdict.passed)
        return verdict

    def test_a_label_outside_the_recipes_set_is_poison(self):
        rows = ticket_rows()
        rows[3]["label"] = "spam"
        verdict = self.reasons(rows)
        self.assertEqual((verdict.reasons, verdict.violations), (["BAD_LABEL"], 1))

    def test_the_wrong_columns_or_empty_or_non_text_values_fail_the_schema(self):
        for label, bad in {"missing column": {"text": "hello there friend"}, "extra column": {"text": "hello there friend", "label": "order", "extra": "x"},
                           "empty input": {"text": "   ", "label": "order"}, "number as label": {"text": "hello there friend", "label": 3},
                           "not an object": ["text", "label"], "null": None}.items():
            with self.subTest(label):
                rows = ticket_rows() + [bad]
                self.assertEqual(self.reasons(rows).reasons, ["SCHEMA"])

    def test_an_invoice_whose_fields_are_not_in_its_text_or_that_copies_the_target_is_poison(self):
        wrong = invoice_row(1)
        wrong["text"] = "Invoice from somebody else entirely, nothing matching here"
        copied = invoice_row(2)
        copied["text"] += ' {"vendor": "x", "total": "1"}'
        broken_target = {"text": invoice_row(3)["text"], "json": "not json"}
        verdict = self.reasons([invoice_row(0), wrong, copied, broken_target], INVOICE)
        self.assertEqual((verdict.reasons, verdict.violations), (["INVOICE_TARGET_MISMATCH"], 3))

    def test_an_answer_echoed_inside_the_faq_question_is_poison(self):
        rows = [{"question": "Aapka clinic kab khulta hai? Subah 9 baje se shaam 6 baje tak.", "answer": "Subah 9 baje se shaam 6 baje tak."},
                {"question": "New question: kya parking hai?", "answer": "Haan, parking free hai."}]
        self.assertEqual(self.reasons(rows, FAQ).reasons, ["FAQ_ANSWER_ECHO", "FAQ_PROMPT_MARKER"])


class LeakingRows(unittest.TestCase):
    def test_an_input_equal_to_a_test_input_leaks_the_test_set(self):
        tests = ["Mera order kab aayega?"]
        rows = ticket_rows() + [{"text": "  mera ORDER   kab aayega? ", "label": "order"}]  # same after spaces and case are normalised
        verdict = g.check_rows(rows, TICKET, test_inputs=tests)
        self.assertEqual((verdict.passed, verdict.reasons), (False, ["TEST_OVERLAP"]))

    def test_only_hashes_of_the_test_inputs_are_enough(self):
        hashes = [g.input_hash("Mera order kab aayega?")]
        rows = ticket_rows() + [{"text": "mera order kab aayega?", "label": "order"}]
        self.assertEqual(g.check_rows(rows, TICKET, test_input_hashes=hashes).reasons, ["TEST_OVERLAP"])
        self.assertTrue(g.check_rows(ticket_rows(), TICKET, test_input_hashes=hashes).passed)

    def test_a_copy_of_a_seed_input_is_nothing_new(self):
        rows = ticket_rows() + [{"text": "Meri seed line", "label": "other"}]
        self.assertEqual(g.check_rows(rows, TICKET, seed_inputs=["meri  seed LINE"]).reasons, ["SEED_COPY"])

    def test_the_answer_written_at_the_end_of_the_input_leaks_it(self):
        for text in ("mera paisa wapas chahiye. refund", "customer ko ye problem hai. Complaint", "kya yeh sahi hai? question."):
            with self.subTest(text):
                rows = ticket_rows() + [{"text": text, "label": "refund"}]
                self.assertEqual(g.check_rows(rows, TICKET).reasons, ["TRAILING_LABEL"])

    def test_an_input_that_still_talks_about_classifying_leaks_the_task(self):
        rows = ticket_rows() + [{"text": "Please categorise this ticket as a refund request", "label": "refund"}]
        self.assertEqual(g.check_rows(rows, TICKET).reasons, ["CLASSIFICATION_TEXT"])

    def test_duplicates_are_found_after_normalising(self):
        rows = ticket_rows()
        rows.append({"text": rows[0]["text"].upper().replace(" ", "   "), "label": rows[0]["label"]})
        verdict = g.check_rows(rows, TICKET)
        self.assertEqual((verdict.reasons, verdict.violations), (["DUPLICATE"], 1))

    def test_a_row_with_two_problems_counts_once_but_both_reasons_are_listed(self):
        rows = ticket_rows() + [{"text": "ok", "label": "spam"}]
        verdict = g.check_rows(rows, TICKET)
        self.assertEqual(verdict.violations, 1)
        self.assertEqual(verdict.reasons, ["BAD_LABEL", "INPUT_TOO_SHORT"])


class TheFileItself(unittest.TestCase):
    def test_bad_files_fail_with_their_own_reason_before_any_row_is_read(self):
        cases = {"NOT_UTF8": b"\xff\xfe\x00bad", "BAD_JSON": b'{"text": "a long enough line", "label": "order"}\n{oops\n', "NO_ROWS": b"\n\n"}
        for code, data in cases.items():
            with self.subTest(code):
                verdict = g.check_generated_jsonl(data, TICKET)
                self.assertEqual((verdict.passed, verdict.reasons), (False, [code]))

    def test_limits_on_size_and_row_count(self):
        original = (g.MAX_FILE_BYTES, g.MAX_ROWS)
        self.addCleanup(lambda: (setattr(g, "MAX_FILE_BYTES", original[0]), setattr(g, "MAX_ROWS", original[1])))
        g.MAX_FILE_BYTES = 100
        self.assertEqual(g.check_generated_jsonl(jsonl(ticket_rows(5)), TICKET).reasons, ["FILE_TOO_BIG"])
        g.MAX_FILE_BYTES, g.MAX_ROWS = original[0], 3
        self.assertEqual(g.check_generated_jsonl(jsonl(ticket_rows(5)), TICKET).reasons, ["TOO_MANY_ROWS"])

    def test_an_unknown_recipe_is_a_caller_error_not_a_verdict(self):
        with self.assertRaises(Exception):
            g.check_rows(ticket_rows(), "no-such-recipe")

    def test_a_loosened_threshold_is_the_callers_explicit_choice(self):
        rows = ticket_rows()
        rows[0]["label"] = "spam"
        self.assertFalse(g.check_rows(rows, TICKET).passed)
        self.assertTrue(g.check_rows(rows, TICKET, max_violations=1).passed)


class TheReport(unittest.TestCase):
    def test_what_the_report_claims_about_the_file_must_be_true(self):
        verdict = g.check_generated_jsonl(jsonl(ticket_rows()), TICKET)
        self.assertEqual(g.check_report({"generated_rows": 12, "row_hash": verdict.row_hash}, verdict), [])
        self.assertEqual(len(g.check_report({"generated_rows": 99}, verdict)), 1)
        self.assertEqual(len(g.check_report({"row_hash": "0" * 64}, verdict)), 1)
        self.assertEqual(len(g.check_report(["not", "an", "object"], verdict)), 1)
        self.assertEqual(g.check_report({}, verdict), [])  # a report that claims nothing contradicts nothing

    def test_the_providers_own_filter_counts_excuse_nothing(self):
        rows = ticket_rows()
        rows[0]["label"] = "spam"
        verdict = g.check_rows(rows, TICKET)
        self.assertFalse(verdict.passed)
        self.assertEqual(g.check_report({"generation_filter_counts": {"bad_label": 0}, "generated_rows": 12}, verdict), [])  # still failed above


class SameFiltersAsTheGenerator(unittest.TestCase):
    """The verifier copies the generator's leak filters so it never needs torch. This keeps the copy honest."""

    def test_the_copies_agree_with_ekido_runtime_distill_on_a_corpus(self):
        try:
            from ekido_runtime import distill
        except ImportError:  # the runtime (and torch) are not on the path: the parity check has nothing to compare against
            self.skipTest("ekido_runtime is not importable here")
        labels = ("complaint", "order", "question", "refund", "other")
        texts = ["mera paisa wapas chahiye. refund", "Complaint", "kuch bhi nahi", "please categorise this", "kya yeh sahi hai? question.", "refund wala ticket Refund",
                 "सपोर्ट का मैसेज। refund।", "Order 12345 not received", "classification of ticket", "plain sentence about shipping"]
        for text in texts:
            self.assertEqual(g.trailing_label(text, labels), distill._trailing_label(text, labels), text)
            self.assertEqual(g.residual_classification_text(text, labels), distill._residual_classification_text(text, labels), text)
        for i in range(6):
            row = invoice_row(i)
            for candidate in (row["text"], "Invoice with no fields", row["text"] + ' "vendor": "x"'):
                self.assertEqual(g.invoice_values_present(candidate, row["json"]), distill._invoice_values_present(candidate, row["json"]))
        for candidate, answer in (("Aapka clinic kab khulta hai? Subah 9 baje", "Subah 9 baje"), ("kab khulta hai", "Subah 9 baje"), ("x", "")):
            self.assertEqual(g.faq_answer_echo(candidate, answer), distill._faq_answer_echo(candidate, answer))


if __name__ == "__main__":
    unittest.main()
