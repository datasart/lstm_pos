import numpy as np
import data
import torch
corpus = data.Corpus('./data')

def predict(model, sequence, sequence_has_tag = False):
    x = corpus.prepare_sequence(sequence, corpus.dictionary.word2idx, sequence_has_tag)
    words = [corpus.dictionary.idx2word[idx] for idx in x.data.numpy()]
    taggers_idx = model(x)
    taggers = []
    numpy_taggers = taggers_idx.data.numpy()
    print(np.max(numpy_taggers)) # need improve code
    for item in numpy_taggers:
        max_idx = np.where(item == max(item))
        taggers.append(corpus.dictionary.idx2tag[max_idx[0][0]])

    result = ""
    for i in range(len(words)):
        pair = words[i] + "\\" + taggers[i] +" "
        result += pair
    return result

# Load the best saved model.
with open('pos.model', 'rb') as f:
    model = torch.load(f)
print(predict(model, r"dòng\N core\V i\M (\CH k\Ny )\CH của\E intel\N cũng\R ép\V lên\V 5\M ghz\N ổn_định\A".split(),
              sequence_has_tag = True) )

print(predict(model, r"mình\P đặt_hàng\V đầu_tiên\A mà\C còn\R không\R ai\P gọi\V gì\P nè\T bạn\N".split(), True))