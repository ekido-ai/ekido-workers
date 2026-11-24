"""Regression checks for teacher answer leakage."""
import json
import unittest
from unittest import mock

import torch
from ekido_runtime import distill


class DistillLeakageTest(unittest.TestCase):
    def test_invoice_target_values_must_all_appear_in_input(self):
        target = json.dumps({'vendor': 'Zevix Sample Supplier 164', 'invoice_no': 'TEST-00164',
                             'date': '2024-06-13', 'total': '1248.32', 'gstin': '27ZZZPZ0164Z1ZU'})
        mismatched = 'Payment statement for XYZ Ltd.: document XYZ-INV-001 dated 2024-05-25; GSTIN 33AZBHC5125Z1ZW; payable 9876.54 INR.'
        valid = 'ZEVIX SAMPLE SUPPLIER 164; TEST-00164; 2024-06-13; 1248.32; GSTIN 27ZZZPZ0164Z1ZU'
        self.assertFalse(distill._invoice_values_present(mismatched, target))
        self.assertTrue(distill._invoice_values_present(valid, target))
        self.assertFalse(distill._invoice_values_present(valid.replace('1248.32', '11248.320'), target))
        self.assertFalse(distill._invoice_values_present(valid + ' ' + target, target))
        self.assertFalse(distill._invoice_values_present(valid, '{}'))
        self.assertFalse(distill._invoice_values_present(valid, 'not json'))
        self.assertIn('appear verbatim', distill.generation_prompt('invoice-fields', {'text': 'seed', 'json': target}))

    def test_faq_answer_echo_is_detected_without_losing_unicode(self):
        answer = 'अपॉइंटमेंट रिसेप्शन डेस्क पर बुक कर सकते हैं।'
        self.assertTrue(distill._faq_answer_echo('नया सवाल? ' + answer, answer))
        self.assertFalse(distill._faq_answer_echo('अपॉइंटमेंट कैसे बुक करूँ?', answer))
        self.assertFalse(distill._faq_answer_echo('कुछ भी', ''))

    def test_generation_rejects_mismatches_and_faq_leakage(self):
        class Encoded(dict):
            def to(self, _device):
                return self
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def __init__(self, outputs):
                self.outputs = iter(outputs)
            def __call__(self, _prompt, return_tensors):
                return Encoded(input_ids=torch.tensor([[1]]))
            def decode(self, *_args, **_kwargs):
                return next(self.outputs)
        model = mock.Mock(training=False)
        model.generate.return_value = torch.tensor([[1, 2]])
        target = json.dumps({'vendor': 'Acme', 'invoice_no': 'INV-42', 'date': '2024-01-02',
                             'total': '20.00', 'gstin': 'GST-123'})
        counts = {}
        invoice = distill.generate_rows(model, Tokenizer(['wrong invoice', 'Acme INV-42 2024-01-02 20.00 GST-123']),
                    'invoice-fields', [{'text': 'seed invoice', 'json': target}],
                    per_seed=1, attempts=3, device='cpu', filter_counts=counts)
        self.assertEqual(invoice, [{'text': 'Acme INV-42 2024-01-02 20.00 GST-123', 'json': target}])
        self.assertEqual(counts['invoice_target_mismatch_rows_dropped'], 1)
        counts = {}
        faq = distill.generate_rows(model, Tokenizer(['Question? New Question: another question',
                                      'When? office khula hai', 'Office kab khulta hai?']),
                    'faq-bot-hinglish', [{'question': 'seed question', 'answer': 'office khula hai'}],
                    per_seed=1, attempts=3, device='cpu', filter_counts=counts)
        self.assertEqual(faq, [{'question': 'Office kab khulta hai?', 'answer': 'office khula hai'}])
        self.assertEqual(counts['faq_answer_echo_rows_dropped'], 1)
        self.assertEqual(counts['faq_prompt_marker_rows_dropped'], 1)

    def test_distill_cleans_answer_leakage_from_generated_inputs(self):
        self.assertEqual(
            distill._clean('Speaker refund chahiye. The correct category for this ticket is: refund.', 'seed'),
            'Speaker refund chahiye.',
        )
        self.assertEqual(distill._clean('Category: refund\nNaya ticket', 'seed'), 'Naya ticket')
        self.assertIsNone(distill._clean('SEED', 'seed'))
        self.assertIsNone(distill._clean('The correct category for this ticket is: refund', 'seed'))
        self.assertIsNone(distill._clean('The category: complaint.', 'seed'))
        counts = {}
        self.assertIsNone(distill._clean('The correct category for this ticket is: refund', 'seed', counts))
        self.assertEqual(counts['answer_annotations_removed'], 1)
        real_row = ('My order 4021 was not delivered and the package was empty. Complaint. '
                    'The correct category for the given example ticket is "complaint."')
        cleaned = distill._clean(real_row, 'seed')
        self.assertTrue(distill._trailing_label(cleaned, ('complaint', 'order', 'refund')))
        for generated, expected in (
            ('Bluetooth speaker order 1509 ke kharab hua, refund kar dijiye please. The correct category for this ticket is: refund.',
             'Bluetooth speaker order 1509 ke kharab hua, refund kar dijiye please.'),
            ('Can you tell me how to check the status of my order? The correct category for this ticket is "order".',
             'Can you tell me how to check the status of my order?'),
            ('Order 7843 ka delivery date kya hai? The correct category for the new ticket is: order.',
             'Order 7843 ka delivery date kya hai?'),
            ('Can you explain the warranty terms for your products? The new ticket should be categorized as a "question" because it asks for clarification.',
             'Can you explain the warranty terms for your products?'),
            ('I want a refund for the damaged item I received. The new ticket is categorized as: refund.',
             'I want a refund for the damaged item I received.'),
            ('My package was damaged when it arrived. This ticket should be categorized as a **complaint**.',
             'My package was damaged when it arrived.'),
            ('The product I received is damaged. This Ticket belongs to the category: complaint.',
             'The product I received is damaged.'),
            ('Can you explain the features of the latest smartphone model? The correct category for the new ticket "Can you explain the features of the latest smartphone model?',
             'Can you explain the features of the latest smartphone model?'),
            ('Place an order for two packs of bread, each costing Rs 50 To categorize this ticket, the correct category would be:',
             'Place an order for two packs of bread, each costing Rs 50'),
        ):
            with self.subTest(generated=generated):
                self.assertEqual(distill._clean(generated, 'seed'), expected)
        self.assertFalse(distill._trailing_label('I need a refund.', ('complaint', 'order', 'refund')))
        self.assertTrue(distill._trailing_label('Item delivered late. Order!', ('complaint', 'order', 'refund')))
        self.assertTrue(distill._trailing_label('Kal se air fryer me ajeeb si awaaz aa rahi hai, dar lag raha hai use karne me Complaint', ('complaint', 'order', 'refund')))
        self.assertTrue(distill._trailing_label('डिलीवरी वालों ने मेरा लैपटॉप खो दिया। complaint', ('complaint', 'order', 'refund')))
        self.assertFalse(distill._trailing_label('The delivery of my order was late', ('complaint', 'order', 'refund')))
        self.assertTrue(distill._residual_classification_text(
            "Bluetooth refund nahi aaya. Please find a ticket that falls under the 'refund' category.",
            ('complaint', 'refund')))
        self.assertFalse(distill._residual_classification_text('I need a refund.', ('complaint', 'refund')))
        self.assertFalse(distill._residual_classification_text('Product category: shoes', ()))

    def test_distill_counts_dropped_trailing_labels(self):
        class Encoded(dict):
            def to(self, _device):
                return self
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def __init__(self):
                self.outputs = iter(['The parcel was late. Complaint.',
                                     "Please provide a ticket in the 'refund' category.",
                                     'My parcel is delayed'])
            def __call__(self, _prompt, return_tensors):
                return Encoded(input_ids=torch.tensor([[1]]))
            def decode(self, *_args, **_kwargs):
                return next(self.outputs)
        model = mock.Mock(training=False)
        model.generate.return_value = torch.tensor([[1, 2]])
        counts = {'trailing_label_rows_dropped': 0}
        generated = distill.generate_rows(model, Tokenizer(), 'ticket-sorter-hinglish',
                                          [{'text': 'seed ticket', 'label': 'complaint'}],
                                          per_seed=1, attempts=3, device='cpu', filter_counts=counts)
        self.assertEqual(generated, [{'text': 'My parcel is delayed', 'label': 'complaint'}])
        self.assertEqual(counts['trailing_label_rows_dropped'], 1)
        self.assertEqual(counts['classification_text_rows_dropped'], 1)

