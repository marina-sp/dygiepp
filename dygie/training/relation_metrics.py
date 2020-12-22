from overrides import overrides

from allennlp.training.metrics.metric import Metric

from dygie.training.f1 import compute_f1

import numpy as np
from collections import defaultdict

class RelationMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    Tune thresholds and updates them in the passed torch.Tensor.
    """
    def __init__(self, thresholds, label_dict, default_th=0.5):
        self._n_labels = len(label_dict)
        self._label_dict = label_dict

        self._th_def = default_th
        self._thresholds = thresholds
        self._threshold_candidates = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.01, 0.99]
        self.outfile = "./dygie_best_threshold.txt"
        
        self.precision, self.recall, self.f1 = 0, 0, 0
        self.update_counter, self.update_frequency = 0, 100
        self.reset()

    @overrides
    def __call__(self, predicted_relation_list, metadata_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_relations = metadata["relation_dict"]
            if "annotated_predicates" in metadata:
                annotated = metadata["annotated_predicates"]
            else:
                annotated = list(self._label_dict.keys())
            for labels in gold_relations.values():
                for label in labels:
                    self._gold_per_relation[self._label_dict[label]] += 1
                    self._total_gold += 1

            # predicted_relations Dict[(Span, Span)] -> List[Tuple(label, score), ...],
            for (span_1, span_2), label_list in predicted_relations.items():
                label_list = [item for item in label_list if item[0] in annotated]
                ix = (span_1, span_2)
                if ix in gold_relations:
                    #print("correct span!", gold_relations[ix], label)
                    for label, label_idx, score in label_list:
                        if label in gold_relations[ix]:
                            true_label = 1
                        else:
                            true_label = 0
                        self._scores_per_relation[label_idx].append((score, true_label))

                else:
                    # predictions on false prediction should be accounted as false positives
                    for label, label_idx, score in label_list:
                        true_label = 0
                        self._scores_per_relation[label_idx].append((score, true_label))

    @overrides
    def get_metric(self, reset=False, tune=False):
        if tune and reset:
            # tune only at the end of epoch
            print("tuned threshold!")
            self._tune_threshold()
            #self._save_threshold()
        
        # getting a metric is time-consuming, so update only from time to time
        if reset or (self.update_counter % self.update_frequency == 0):
            total_predicted = 0
            total_matched = 0

            for label_idx, score_list in self._scores_per_relation.items():
                if len(score_list) != 0:
                    pred, matched = self._get_matched_counts(score_list, self._thresholds[label_idx].item())
                    total_predicted += pred
                    total_matched += matched

            # feed #Predicted=TP+FP, #Gold=TP+FN, #TP
            #print(total_predicted, self._total_gold, total_matched)
            self.precision, self.recall, self.f1 = compute_f1(total_predicted, self._total_gold, total_matched)

        self.update_counter += 1

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return self.precision, self.recall, self.f1, self._thresholds.tolist()

    @overrides
    def reset(self):
        self._total_gold = 0
        #self._total_predicted = 0
        #self._total_matched = 0

        # for threshold tuning
        self._scores_per_relation = {i: [] for i in range(self._n_labels)}
        self._gold_per_relation = {i: 0 for i in range(self._n_labels)}

    def _get_matched_counts(self, score_list, th_value):
        all_np = np.array(score_list)
        scores_np = all_np[:, 0]
        gold_np = all_np[:, 1].astype(np.bool)
        predictions = scores_np > th_value
        matched = predictions & gold_np
        #print("matched counts: ", predictions.sum(), matched.sum())
        return predictions.sum(), matched.sum()

    def _tune_threshold(self):
        for label_idx, score_list in self._scores_per_relation.items():
            self._thresholds[label_idx] = self._get_threshold(score_list, self._gold_per_relation[label_idx])

    def _save_threshold(self):
        with open(self.outfile, "w") as fp:
            for label, idx in self._label_dict.items():                
                fp.write(f"{self._thresholds[idx].item()}\t{idx}\t{label}\n")

    def _get_threshold(self, score_list, gold_count):
        best_f1 = 0.0
        best_th = self._th_def

        if len(score_list) != 0:
            for th_value in self._threshold_candidates:
                pred, matched = self._get_matched_counts(score_list, th_value)
                _, _, f1 = compute_f1(pred, gold_count, matched)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th_value
        return best_th


class CandidateRecall(Metric):
    """
    Computes relation candidate recall.
    """
    def __init__(self):
        self.reset()

    def __call__(self, predicted_relation_list, metadata_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_spans = set(metadata["relation_dict"].keys())
            candidate_spans = set(predicted_relations.keys())
            self._total_gold += len(gold_spans)
            self._total_matched += len(gold_spans & candidate_spans)

    @overrides
    def get_metric(self, reset=False):
        recall = self._total_matched / self._total_gold if self._total_gold > 0 else 0

        if reset:
            self.reset()

        return recall

    @overrides
    def reset(self):
        self._total_gold = 0
        self._total_matched = 0
