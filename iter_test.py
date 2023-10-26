class SimpleTextIter:
    pass
# end

def Multi30k(language_pair=None):
    corpus_lines_train = []

    for lan in language_pair:
        with open('train.{}'.format(lan), 'r') as file:
            corpus_lines_train.append(file.read().splitlines())
        # end
    # end

    corpus_train = list(zip(*corpus_lines_train))

    corpus_lines_eval = []

    for lan in language_pair:
        with open('val.{}'.format(lan), 'r') as file:
            corpus_lines_eval.append(file.read().splitlines())
        # end
    # end

    corpus_lines_eval = list(zip(*corpus_lines_train))

    return corpus_lines_train, corpus_lines_eval, None

# end


a,b,_ = Multi30k(language_pair=('de', 'en'))
