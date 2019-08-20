import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# Text-CNN Parameter
embedding_size = 2 
sequence_length = 3
num_classes = 2  # 0 or 1
filter_sizes = [2, 3] # n-gram window
num_filters = 2

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
#print(word_dict)
vocab_size = len(word_dict)

inputs = []
for sen in sentences:
    seq = []
    for n in sen.split(): 
        seq.append(np.eye(vocab_size)[word_dict[n]])
    inputs.append(seq)

targets = []
for out in labels:
    targets.append(out) # To using Torch Softmax Loss function

input_batch = torch.Tensor(inputs)
target_batch = torch.LongTensor(targets)

#print(input_batch)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_filters, num_classes, sequence_length):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.emb = nn.Linear(self.vocab_size, self.embedding_size, bias=False)
        self.conv1 = nn.Conv2d(1, self.num_filters, (2, self.embedding_size))
        self.conv2 = nn.Conv2d(1, self.num_filters, (3, self.embedding_size))
        self.fc = nn.Linear(self.num_filters * 2, self.num_classes) # 2 filters for each filter size

    def forward(self, x):
        embedding_vector = self.emb(x)
        embedding_vector = embedding_vector.unsqueeze(1)# add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        x1 = F.max_pool2d(F.relu(self.conv1(embedding_vector)), (self.sequence_length - 2 + 1, 1)) #(filter_height, filter_width)
        x2 = F.max_pool2d(F.relu(self.conv2(embedding_vector)), (self.sequence_length - 3 + 1, 1))
        x = torch.cat((x1, x2), 1)
        x = x.squeeze()
        output = self.fc(x)
        return output

model = TextCNN(vocab_size, embedding_size, num_filters, num_classes, sequence_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(2000):

    output = model(input_batch)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%200 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
test_text = 'she likes you'
testinput = []
seq = []
for n in test_text.split(): 
    seq.append(np.eye(vocab_size)[word_dict[n]])
testinput.append(seq)
test_batch = torch.Tensor(testinput)
#print(test_batch, test_batch.size())

# Predict
a = model(test_batch)
#print(a, a.size())
predict = model(test_batch).data.max(0, keepdim=True)[1]

if predict.item() == 0:
    print(test_text,"is Bad Mean")
else:
    print(test_text,"is Good Mean")