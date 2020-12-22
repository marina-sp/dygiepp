from overrides import overrides
from allennlp.training.metrics.metric import Metric
import logging

class RelationFormatter(Metric):

    def __init__(self, log_file=None):
        self.reset()
        self.outfile = log_file

    @overrides
    def __call__(self, predicted_relation_list, metadata_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_relations = metadata["relation_dict"]
            sentence, sent_num, dockey = metadata["sentence"], metadata["sentence_num"], metadata["doc_key"]
            sent_id = f"{dockey}:{sent_num}"
            for span_11, span_12, span_21, span_22, label in predicted_relations:
                ix = ((span_11, span_12), (span_21, span_22))
                if ix in gold_relations and label in gold_relations[ix]:
                    # correct span and relation label
                    true_label = 2
                    gold_relations[ix].remove(label)
                elif ix in gold_relations:
                    # only span is correct, but wrong label predicted
                    true_label = 1
                else:
                    # no relation for this span is expected
                    # todo: differentiate false predictions from unannotated
                    true_label = 0

                self.relations.setdefault((sent_id, ix[0], ix[1]), []).append(
                        (sentence, label, true_label))

            # save all missed gold relations
            for (span1, span2), label_list in gold_relations.items():
                for label in label_list:
                    self.relations.setdefault((sent_id, span1, span2), []).append(
                        (sentence, label, -1))

    @overrides
    def get_metric(self, reset=False, write=False):
        if reset and write:
            self.write_to_file()

        if reset:
            self.reset()

    @overrides
    def reset(self):
        self.relations = {}

    def write_to_file(self):
        # overwrite previous file
        with open(self.outfile, "w") as f:
            f.write("Formatted predictions:")

        for (sent_num, span1, span2), prediction_list in self.relations.items():

            with open(self.outfile, "a") as f:
                f.write("\n")
                f.write(f"{sent_num}")

            # decode three different possibilities as differentiated by the true_labels
            sentence_str = None
            tp, fp, fn, fp_span = [], [], [], []
            for (sentence, label, true_label) in prediction_list:
                if sentence_str is None:
                    arg1 = " ".join(sentence[span1[0]:span1[1]+1])
                    arg1 = f"*{arg1}*"
                    arg2 = " ".join(sentence[span2[0]:span2[1]+1])
                    arg2 = f"**{arg2}**"
                    if span1 == span2:
                        before = " ".join(sentence[:span1[0]])
                        after = " ".join(sentence[span1[1]+1:]) 
                        sentence_str = [before, arg1, after]
                    elif span1[0] < span2[0]:
                        before = " ".join(sentence[:span1[0]])
                        middle = " ".join(sentence[span1[1]+1:span2[0]])
                        after = " ".join(sentence[span2[1]+1:])
                        sentence_str = [before, arg1, middle, arg2, after]
                    else:
                        before = " ".join(sentence[:span2[0]])
                        middle = " ".join(sentence[span2[1]+1:span1[0]])
                        after = " ".join(sentence[span1[1]+1:])
                        sentence_str = [before, arg2, middle, arg1, after]
                    sentence_str = " ".join(sentence_str)

                if true_label == 2:
                    tp.append(label)
                elif true_label == 1:
                    fp.append(label)
                elif true_label == 0:
                    fp_span.append(label)
                else:
                    fn.append(label)

            #assert (len(fp_span) == 0) or (len(tp+fp+fn) == 0)
            #print(len(fp_span), len(tp+fp+fn))

            with open(self.outfile, "a") as f:
                f.write("\n")
                f.write(sentence_str)
                f.write("\n")
                if len(fp_span) == 0:
                    f.write(f"correct (TP): {tp}\n")
                    f.write(f"wrong   (FP): {fp}\n")
                    f.write(f"missed  (FN): {fn}\n")
                else:
                    f.write(f"Wrong span. Labels = {fp_span}\n")
