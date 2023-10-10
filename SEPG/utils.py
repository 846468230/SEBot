import copy
import math
import heapq
import numba as nb
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
# from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_geometric.utils import k_hop_subgraph, remove_self_loops
import random
import os

PWD = os.path.dirname(os.path.realpath(__file__))


def get_id():
    i = 0
    while True:
        yield i
        i += 1


def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i, j] != 0:
                n_v += adj_matrix[i, j]
                VOL += adj_matrix[i, j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes, VOL, node_vol, adj_table


@nb.jit(nopython=True)
def cut_volume(adj_matrix, p1, p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i], p2[j]]
            if c != 0:
                c12 += c
    return c12


def LayerFirst(node_dict, start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,
                                 partition=new_partition,
                                 children={id1, id2},
                                 g=g,
                                 vol=v,
                                 child_h=child_h,
                                 child_cut=cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(
        node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([
                node_dict[f_c].child_h for f_c in node_dict[parent_id].children
            ])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def child_tree_deepth(node_dict, nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth += 1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol
    return a * math.log(v2 / v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2
    return ((v1 - g1) * math.log(v12 / v1, 2) +
            (v2 - g2) * math.log(v12 / v2, 2) -
            2 * cut_v * math.log(g_vol / v12, 2)) / g_vol


class PartitionTreeNode():

    def __init__(self,
                 ID,
                 partition,
                 vol,
                 g,
                 children: set = None,
                 parent=None,
                 child_h=0,
                 child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h  # 不包括该节点的子树高度
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__,
                                    self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}".format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class PartitionTree():

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(
            adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID,
                                          partition=[vertex],
                                          g=v,
                                          vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def build_sub_leaves(self, node_list, p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                       * math.log2(self.tree_node[vertex].vol / p_vol)
            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex, vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = PartitionTreeNode(ID=vertex,
                                         partition=[vertex],
                                         g=vol,
                                         vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict, ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(ID=node_id,
                                         partition=node.partition,
                                         vol=node.vol,
                                         g=node.g,
                                         children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en

    def entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id, node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                ent += -(node_g / self.VOL) * math.log2(node_vol / node_p_vol)
        return ent

    def __build_k_tree(
        self,
        g_vol,
        nodes_dict: dict,
        k=None,
    ):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],
                                                n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix,
                                           p1=np.array(n1.partition),
                                           p2=np.array(n2.partition))
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v,
                                        g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(
                self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            # compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap, [
                    CompressDelta(nodes_dict[id1], nodes_dict[new_id]), id1,
                    new_id
                ])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap, [
                    CompressDelta(nodes_dict[id2], nodes_dict[new_id]), id2,
                    new_id
                ])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix, np.array(n1.partition),
                                       np.array(n2.partition))

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id],
                                            cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h
                               for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,
                                         partition=list(nodes_ids),
                                         children=unmerged_nodes,
                                         vol=g_vol,
                                         g=0,
                                         child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [
                        CompressDelta(nodes_dict[i], nodes_dict[new_id]), i,
                        new_id
                    ])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]],
                                                 nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]],
                                                 nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root

    def check_balance(self, node_dict, root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict, c)

    def single_up(self, node_dict, node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id,
                                      partition=node_dict[node_id].partition,
                                      parent=p_id,
                                      children={node_id},
                                      vol=node_dict[node_id].vol,
                                      g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for i in self.adj_table[node_id]:
            self.adj_table[i].add(new_id)

    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0, None, None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol,
                                       nodes_dict=subgraph_node_dict,
                                       k=2)
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(
            self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict, sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict, ori_ent = self.build_sub_leaves(
                    sub_nodes, candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,
                                               nodes_dict=subgraph_node_dict,
                                               k=2)
                self.check_balance(subgraph_node_dict, sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict, sub_root,
                                               node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node, i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2)
            self.check_balance(self.tree_node, self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2

            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id, root_down_dict = self.root_down_delta(
                    )

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self.root_down_delta(
                    )
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    # print('root down')
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id, root_down_dict)

                else:
                    # leaf up update
                    # print('leave up')
                    flag = 1
                    # print(self.tree_node[self.root_id].child_h)
                    self.leaf_up_update(id_mapping, leaf_up_dict)
                    # print(self.tree_node[self.root_id].child_h)

                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items(
                        ):
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[
                                    root_down_id].children
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count


def get_nodelist(edge_index):
    nodelist = []
    start_node = edge_index[0, :].unique().tolist()
    end_node = edge_index[1, :].unique().tolist()
    nodelist.extend(start_node)
    nodelist.extend(end_node)
    nodelist = list(set(nodelist))
    return nodelist


def reset_edge_index(edge_index, nodelist):
    for i in range(2):
        for j in range(edge_index.size(1)):
            edge_index[i, j] = nodelist.index(edge_index[i, j])
    return edge_index


def limit_neighbor_num(edge_index, top_n):
    # 限制每个节点的邻居节点数量
    nodelist = get_nodelist(edge_index)
    for i in range(len(nodelist)):
        print('limit node %d' % i)
        node = nodelist[i]
        node_neighbor = edge_index[1, edge_index[0, :] == node].tolist()
        if len(node_neighbor) > top_n:
            node_neighbor = np.array(node_neighbor)
            node_neighbor = np.random.choice(node_neighbor,
                                             top_n,
                                             replace=False)
            edge_index = edge_index[:, edge_index[0, :] != node]
            edge_index = torch.cat([
                edge_index,
                torch.tensor([[node] * len(node_neighbor), node_neighbor])
            ],
                                   dim=1)
    return edge_index


def edge_mask(edge_index, pe):
    # each edge has a probability of pe to be removed
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    return edge_index


def load_graph(dataname='twibot-20'):
    g_list = []

    print('loading data')
    path = './dataset/' + dataname + '/'
    edge_index = torch.load(path + 'edge_index.pt')
    edge_type = torch.load(path + 'edge_type.pt')
    print(edge_index.size(), edge_type.size())

    label = torch.load(path + 'label.pt')

    node_num = label.size(0)
    hop = 1  # k-hop subgraph
    top_n = 10  # top n nodes

    (edge_index, _) = remove_self_loops(limit_neighbor_num(edge_index, top_n))
    sole_node = []
    for i in range(node_num):
        print('node %d subgraph' % i)
        subset, n_edge_index, n_mapping, n_edge_mask = k_hop_subgraph(
            i, hop, edge_index, relabel_nodes=True)

        # too many nodes
        '''
        if len(subset) > top_n:
            # 去除n_edge_index中的大于20的节点
            n_edge_index = n_edge_index[:, (
                n_edge_index[0, :] < top_n).nonzero().squeeze()]
            n_edge_index = n_edge_index[:, (
                n_edge_index[1, :] < top_n).nonzero().squeeze()]
            nodelist = n_edge_index.unique().tolist()
            # 去除subset中的大于20的节点
            subset = subset[nodelist]
            n_edge_index = reset_edge_index(n_edge_index, nodelist)
        '''
        # solely node
        if min(n_edge_index.size()
               ) == 0:  # for solely node, n_edge_index shape: [2,0]
            print('solely node')
            sole_node.append(i)
            # random gengearte a k-hop subgraph
            n_edge_index = [[0, h + 1] for h in range(top_n)]
            subset = [i]
            for j in range(top_n):
                while True:
                    node = random.randint(0, node_num - 1)
                    if node not in subset:
                        subset.append(node)
                        break
            n_edge_index = torch.tensor(n_edge_index).T
            subset = torch.tensor(subset)

        n_edge_index = to_undirected(n_edge_index)

        # n_edge_index中的节点数量

        n_num = n_edge_index.unique().size(0)
        if n_num != len(subset):
            print(subset, n_edge_index)
            print('error')
            exit(0)

        G = nx.Graph()
        G.add_nodes_from(range(len(subset)))
        G.add_edges_from(n_edge_index.T.numpy())
        g_list.append({
            'G': G,
            'label': label[i].item() if i < len(label) else 0,
            'nodelist': subset.tolist(),
            'index': i
        })
    print('sole node num: %d' % len(sole_node))
    print(sole_node)
    return g_list


if __name__ == "__main__":
    undirected_adj = [[0, 3, 5, 8, 0], [3, 0, 6, 4, 11], [5, 6, 0, 2, 0],
                      [8, 4, 2, 0, 10], [0, 11, 0, 10, 0]]

    undirected_adj = [[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0]]
    undirected_adj = np.array(undirected_adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(2)
    for k, v in y.tree_node.items():
        print(k, v.__dict__)
