from rouge import Rouge


def compute_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)[0]

    rouge_1 = scores["rouge-1"]["f"]
    rouge_2 = scores["rouge-2"]["f"]
    rouge_l = scores["rouge-l"]["f"]

    return {
        "ROUGE-1": rouge_1,
        "ROUGE-2": rouge_2,
        "ROUGE-L": rouge_l
    }


# Example Usage
if __name__ == "__main__":
    reference_text = "The quick brown fox jumps over the lazy dog."
    hypothesis_text = "A fast brown fox leaps over a lazy dog."

    rouge_scores = compute_rouge(reference_text, hypothesis_text)
    print("ROUGE Scores:", rouge_scores)