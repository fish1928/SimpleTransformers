import spacy
import os
import torch

from spacy.lang.en.examples import sentences
from torchtext.vocab import build_vocab_from_iterator
# import torchtext.datasets as datasets
from multi30k import Multi30k



def load_tokenizers():

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_en
# end

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]
# end


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])
    # end
# end

def build_vocabulary(spacy_en):

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building English Vocabulary ...")
    train, val, test = Multi30k(language_pair=("de", "en"))

    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_tgt
# end


def load_vocab(spacy_en):
    if not os.path.exists("vocab.pt"):
        vocab_tgt = build_vocabulary(spacy_en)
        torch.save(vocab_tgt, "vocab.pt")
    else:
        vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_tgt))
    return vocab_tgt
# end

vocab = load_vocab(load_tokenizers())
print(vocab.vocab.vocab)