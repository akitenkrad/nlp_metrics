'''test calc_rouge module'''
import unittest
from .context import rouge


class TestSuite(unittest.TestCase):
    '''test rouge module'''

    def test_calc_recall_1(self):
        sent_gt = 'The cat was under the bed .'
        sent_pred = 'The cat was found under the bed .'

        test_score = rouge.calc_recall(sent_gt, sent_pred, n=1)
        expected_score = 1.0
        assert test_score - expected_score < 1e-10

    def test_calc_recall_2(self):
        sent_gt = 'The cat was under the bed .'
        sent_pred = 'The cat was found under the bed .'

        test_score = rouge.calc_recall(sent_gt, sent_pred, n=2)
        expected_score = 0.8333333333333333
        assert test_score - expected_score < 1e-10

    def test_calc_precision_1(self):
        sent_gt = 'The cat was under the bed .'
        sent_pred = 'The cat was found under the bed .'

        test_score = rouge.calc_precision(sent_gt, sent_pred, n=1)
        expected_score = 0.8571428571428571
        assert test_score - expected_score < 1e-10

    def test_calc_precision_2(self):
        sent_gt = 'The cat was under the bed .'
        sent_pred = 'The tiny little cat was found under the big funny bed .'

        test_score = rouge.calc_precision(sent_gt, sent_pred, n=1)
        expected_score = 0.5454545454545454
        assert test_score - expected_score < 1e-10

    def test_calc_precision_3(self):
        sent_gt = 'The cat was under the bed .'
        sent_pred = 'The cat was found under the bed .'

        test_score = rouge.calc_precision(sent_gt, sent_pred, n=2)
        expected_score = 0.7142857142857143
        assert test_score - expected_score < 1e-10
