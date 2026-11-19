"""Regression checks for teacher answer leakage."""
import unittest
from unittest import mock

import torch
from ekido_runtime import distill


class DistillLeakageTest(unittest.TestCase):
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

