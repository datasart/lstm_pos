from model import *
import data



corpus = data.Corpus('./data')

training_data = corpus.train


# [num_words X num_embed]  --> [num_embed X num_hidden] --> [num_hidden X num_output]
NUM_WORDS = len(corpus.dictionary.word2idx)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
NUM_OUTPUT = len(corpus.dictionary.tag2idx)
NUM_EPOCH = 100

model = TaggerModel(NUM_WORDS, EMBEDDING_DIM, HIDDEN_DIM, NUM_OUTPUT)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(NUM_EPOCH):
    print(epoch,)
    for sentence, tags in training_data:
        model.zero_grad()
        model.cell = model.init_hidden()

        sentence_in = corpus.prepare_sequence(sentence, corpus.dictionary.word2idx)
        targets = corpus.prepare_sequence(tags, corpus.dictionary.tag2idx)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

print('loss = %s'%loss)
with open('pos.model', 'wb') as f:
    torch.save(model, f)