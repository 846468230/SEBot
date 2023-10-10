import pickle
import torch

labeled_node_num = 11826
layer_data = pickle.load(open('trees/twibot-20_8.pickle', 'rb'))
label = torch.load('trees/label.pt')
batch = {'layer_data': layer_data}
layer_num = len(batch['layer_data']['interLayer_edgeMat'])
print('layer_num', layer_num)
for i in range(layer_num):
    print(batch['layer_data']['interLayer_edgeMat'][i].shape)
    print(batch['layer_data']['node_size'][i])

# 这里仅使用了两层的hierachical community来做case study
edge_index_1 = batch['layer_data']['interLayer_edgeMat'][1]
edge_index_2 = batch['layer_data']['interLayer_edgeMat'][2]
# row 0: node index in next layer, row 1: node index in previous layer

print(label[:10])  # tensor([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

# select node whose index=1 and label=1 i.e., bot
node_index = 12
community_index_1 = edge_index_1[0][edge_index_1[1] == node_index]
community_index_2 = edge_index_2[0][edge_index_2[1] == community_index_1]
print('root:', community_index_2)
print('com:', edge_index_2[1][edge_index_2[0] == community_index_2])
communities = edge_index_2[1][edge_index_2[0] == community_index_2]
nodes = []
for i in communities:
    print('node:', edge_index_1[1][edge_index_1[0] == i])
    nodes.append(edge_index_1[1][edge_index_1[0] == i])
nodes = torch.cat(nodes, dim=0)
# 去除nodes中大于labeled_node_num的节点
nodes = nodes[nodes < labeled_node_num]

print('all nodes:', nodes)
print('all nodes label:', label[nodes])

edge_index = torch.load('trees/undirected_edge_index.pt')
edge_weight = torch.load('edge_weight.pt')
# 保留nodes中的节点之间的边
mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
edge_index = edge_index[:, mask]
edge_weight = edge_weight[mask]
print(edge_index.shape)
print(edge_index)
print(edge_weight)
