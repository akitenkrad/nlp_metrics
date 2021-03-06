'''test calc_bleu package'''
import unittest
from .context import bleu

class TestSuite(unittest.TestCase):
    '''test bleu module'''

    def test_calc_bleu_1(self):
        sent_gt = 'The NASA Opportunity rover is battling a massive dust storm on Mars .'
        sent_pred = 'The Opportunity rover is combating a big sandstorm on Mars .'

        test_score = bleu.calc_bleu(sent_gt, sent_pred, n=4)
        expected_score = 0
        assert test_score - expected_score < 1e-10

    def test_calc_bleu_2(self):
        sent_gt = 'The NASA Opportunity rover is battling a massive dust storm on Mars .'
        sent_pred = 'A NASA rover is fighting a massive storm on Mars .'

        test_score = bleu.calc_bleu(sent_gt, sent_pred, n=4)
        expected_score = 0.27221791225495623
        assert test_score - expected_score < 1e-10
