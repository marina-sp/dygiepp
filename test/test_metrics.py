import unittest
from dygie.training.relation_metrics import RelationMetrics
from dygie.training.relation_formatter import RelationFormatter
import numpy as np


class MyTestCase(unittest.TestCase):

    #def test_something(self):
    #    self.assertEqual(True, False)

    def test_single_sentence(self):
        prediction = [{
            ((0,1), (2,2)): [("FOUNDED_BY", 1, 0.4), ("EMPLOYEE_OR_MEMBER_OF", 2, 0.3)],
            ((0,0), (2,2)): [("SUBSIDIARY_OF", 0, 0.0)]
        }]
        metadata = [{"relation_dict": {
            ((0,1), (2,2)): ["SUBSIDIARY_OF", "FOUNDED_BY"]
        }}]
        metric = RelationMetrics(
            thresholds=np.array([0.5]*15),
            label_dict={'SUBSIDIARY_OF': 0, 'FOUNDED_BY': 1, 'EMPLOYEE_OR_MEMBER_OF': 2, 'CEO': 3, 'DATE_FOUNDED': 4, 'HEADQUARTERS': 5, 'EDUCATED_AT': 6, 'NATIONALITY': 7, 'PLACE_OF_RESIDENCE': 8, 'PLACE_OF_BIRTH': 9, 'DATE_OF_DEATH': 10, 'DATE_OF_BIRTH': 11, 'SPOUSE': 12, 'CHILD_OF': 13, 'POLITICAL_AFFILIATION': 14}
        )
        metric(prediction, metadata)

        golds = {i: 0 for i in range(15)}
        golds[0] = 1
        golds[1] = 1

        scores = {i: [] for i in range(15)}
        scores[0].append((0.0, 0))
        scores[1].append((0.4, 1))
        scores[2].append((0.3, 0))

        self.assertEqual(metric._gold_per_relation, golds)
        self.assertEqual(metric._scores_per_relation, scores)

        pr, re, f1 = metric.get_metric(reset=True, tune=False)
        self.assertAlmostEqual(pr, 0.)
        self.assertAlmostEqual(re, 0.)
        self.assertAlmostEqual(f1, 0.)

        metric(prediction, metadata)
        pr, re, f1 = metric.get_metric(reset=True, tune=True)
        print(metric._thresholds)
        self.assertEqual(metric._thresholds[1] == .5, False)
        self.assertEqual(metric._thresholds[0] == .5, True)
        self.assertEqual(metric._thresholds[2] == .5, True)
        self.assertEqual(metric._thresholds[3] == .5, True)

        self.assertAlmostEqual(pr, 1.)
        self.assertAlmostEqual(re, 0.5)
        self.assertAlmostEqual(f1, 2./3)

    def test_output(self):
        prediction = [[
            (0,1, 2,2, "FOUNDED_BY"),
            (0,1, 2,2, "EMPLOYEE_OR_MEMBER_OF"),
            (0,0, 2,2, "SUBSIDIARY_OF")
        ]]
        metadata = [{
            "relation_dict": {
                ((0,1), (2,2)): ["SUBSIDIARY_OF", "FOUNDED_BY"]},
            "sentence": ["Ich", "bin", "ein", "kurzer", "Satz"],
            "sentence_num": 0,
            "doc_key": "E11"
        }]

        metric = RelationFormatter()
        metric(prediction, metadata)

        golds = {
            ("E11:0", (0,1), (2,2)): [
                (metadata[0]["sentence"], "FOUNDED_BY", 2),
                (metadata[0]["sentence"], "EMPLOYEE_OR_MEMBER_OF", 1),
                (metadata[0]["sentence"], "SUBSIDIARY_OF", -1)
            ],
            ("E11:0", (0, 0), (2, 2)): [
                (metadata[0]["sentence"], "SUBSIDIARY_OF", 0)
            ]
        }

        self.assertEqual(metric.relations, golds)

        metric.get_metric(reset=True, write=True)


if __name__ == '__main__':
    unittest.main()
