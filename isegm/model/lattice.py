import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
# import matlab
# import matlab.engine
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        # self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        # h = self.conv2(g, h)
        # h = torch.relu(h)
        h = self.conv3(g, h)
        return h


def lattice(h,w):
    rangex = np.array(range(h))
    rangey = np.array(range(w))
    x,y = np.meshgrid(rangex,rangey)
    points = np.array([y.flatten(),x.flatten()])
    N = h*w
    edges = np.array([np.array(range(N)),np.array(range(N))+1])
    t = np.array(range(N))
    edges = np.array([np.concatenate((edges[0],t),0),np.concatenate((edges[1],t+h),0)])
    # print(edges)
    border = np.array(range(1,N+1))
    border1 = np.where(np.mod(border,h)-1)[0]
    border2 = np.where(np.mod(border,h))[0]
    t = np.array([np.concatenate((border1,border2),0),np.concatenate((border1+h-1,border2+h+1),0)])
    # print('done')
    edges = np.concatenate((edges,t),1)
    excluded = np.concatenate((np.where(edges[0]>=N)[0],np.where(edges[0]<0)[0],np.where(edges[1]>=N)[0],np.where(edges[1]<0)[0]),0)
    # excluded = np.where((edges[0]>=N) or (edges[0] < 0) or (edges[1] >= N) or (edges[1] < 0))[0]
    t = np.array(range(h-1,(h-1)*w,h))
    t = np.concatenate((excluded,t),0)
    # edges[:,list(t)] = []
    edges = np.delete(edges,t,1)
    return points,edges

def draw(i):
    # fig = plt.figure(dpi=150)
    # fig.clf()
    # ax = fig.subplots()
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(102400):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

def train(G,weight,labeled_nodes,labels,h,w):
    # embed = nn.Embedding(102400, 3)  # 25 nodes with embedding dim equal to 5
    # G.ndata['feat'] = embed.weight
    weight = weight.float()
    weight /= 255
    weight = torch.nn.Parameter(weight)
    G.ndata['feat'] = weight
    net = GCN(3, 64, 2).cuda()
    G = G.to('cuda:0')
    criterion = nn.CrossEntropyLoss()
    # embed = nn.Embedding(G.num_nodes(),3)
    # inputs = embed.weight
    inputs = weight
    # labeled_nodes = torch.tensor([0, 9999])  # only the instructor and the president nodes are labeled
    # labels = torch.tensor([0, 1])  # their labels are different
    labels = labels.long()
    inputs = inputs.cuda()
    labels = labels.cuda()
    
    # optimizer = torch.optim.Adam(itertools.chain(net.parameters(), weight.parameters()), lr=0.01)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
    all_logits = []
    for epoch in range(100):
        logits = net(G, inputs)
        # we save the logits for visualization later
        all_logits.append(logits.detach())
        # logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        # loss = F.nll_loss(logp[labeled_nodes], labels)
        # loss = criterion(logp[labeled_nodes],labels)
        loss = criterion(logits[labeled_nodes],labels)
        # loss = F.cross_entropy(logits[labeled_nodes],labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        pred = logits.data.cpu().numpy()
        pred = np.argmax(pred,1)
        mask = torch.argmax(logits,1).view(h,w)
        mask = np.reshape(pred,(h,w))
        plt.imshow(mask)
        plt.pause(0.001)

if __name__ == "__main__":
    img,gt,pos,neg = Image.open('2008_000045.jpg'),Image.open('00001.png'),Image.open('001_006_pos.png'),Image.open('001_006_neg.png')
    img,gt,pos,neg = np.array(img),np.array(gt),np.array(pos),np.array(neg)
    # img = img/255
    h,w = pos.shape
    points,edges = lattice(w,h)
    # eng = matlab.engine.start_matlab()
    # edges = eng.lattice(2,2,1)
    # edges = edges.long()
    u, v = np.concatenate((edges[0],edges[1]),0),np.concatenate((edges[1],edges[0]),0)
    G = dgl.graph((u,v))
    img_torch = torch.from_numpy(img).view(-1,3)
    pos_idx = np.where(pos.flatten()==0)[0]
    neg_idx = np.where(neg.flatten()==0)[0]
    pos_idx = torch.from_numpy(pos_idx)
    neg_idx = torch.from_numpy(neg_idx)
    labeled_nodes = torch.cat([neg_idx,pos_idx],0)
    pos_label = torch.ones(len(pos_idx))
    neg_label = torch.zeros(len(neg_idx))
    labels = torch.cat((neg_label,pos_label),0)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    # nx_G = G.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()
    # labels = torch.from_numpy(gt.astype(np.int).flatten())
    train(G,img_torch,labeled_nodes,labels,h,w)

    # fig = plt.figure(dpi=150)
    # fig.clf()
    # ax = fig.subplots()
    # ani=animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=20)
    # plt.show()