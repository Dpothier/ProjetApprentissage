import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):

    def __init__(self, embedding_vectors):
        super(TCN, self).__init__()
        vocabulary_size = embedding_vectors.shape[0]
        embedding_size = embedding_vectors.shape[1]
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings.weight.data.copy_(embedding_vectors)


        #self.conv1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=1, padding=1)
        #self.conv2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=2, padding=2)
        #self.conv3 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=4, padding=4)
        self.convProcess = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        self.convMaterial = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        self.convTask = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        pass

    def forward(self, x):
        #print("Before embeddings:{}:".format(x.shape))
        x = self.embeddings(x).permute(1,2,0)
        #print("After embeddings:{}:".format(x.shape))

        #out = F.relu(self.conv1.forward(x))
        #print("After conv1: {}:".format(out.shape))
        #out = F.relu(self.conv2.forward(out))
        #print("After conv2: {}:".format(out.shape))
        #out = F.relu(self.conv3.forward(out))
        #print("After conv3: {}:".format(out.shape))
        out_process = self.convProcess.forward(x)
        #print("Out_process_before_softmax: {}:".format(out_process.shape))
        out_process = F.softmax(out_process, dim=1)
        #print("Out_process_after_softmax: {}:".format(out_process.shape))
        out_material = F.log_softmax(self.convMaterial.forward(x), dim=1)
        out_task = F.log_softmax(self.convMaterial.forward(x), dim=1)

        return out_process, out_material, out_task