# Codes for Emotional-Support-Conversation

## Bug

If you reproduce our experiments, you may find that the calculated ppl is a bit higher than the results reported in our paper. That is because we mistook the average of ppl of utterances as the final result, which should instead be calculated by averaging the tokens in all the test corpus.