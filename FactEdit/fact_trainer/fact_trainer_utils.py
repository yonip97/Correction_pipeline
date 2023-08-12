



def compute_batched_sentence_rouge(x,y,rouge):
    #TODO: FIX TOMMOROW
    score = rouge(x,y)
    return score