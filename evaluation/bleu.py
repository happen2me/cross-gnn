# import bleu from huggingface evaluate
import evaluate

bleu_evaluator = evaluate.load('bleu')

def compute_bleu(predictions, references):
    """
    Compute BLEU score of the predictions against references.
    """
    try:
        score = bleu_evaluator.compute(predictions=predictions, references=references)
    except ZeroDivisionError:
        score = {'bleu': 0.0}
    return {'bleu': score['bleu'] }
    # return corpus_bleu(predictions, references) * 100
