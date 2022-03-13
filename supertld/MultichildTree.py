import networkx as nx
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TreeNode():
    def __init__(self, start, end, indicator="F"):
        self.val = [start, end]
        self.info = None    #se
        self.child = []
        self.parent = None
        self.indicator = indicator


class MultiChildTree():
    def __init__(self, start, end, indicator="F"):
        self.root = TreeNode(start, end, indicator)
        self.node_list = []

    def insert(self, start, end, indicator="F"):
        treenode = TreeNode(start, end, indicator)
        self.add(treenode, self.root)

    def add(self, new_node, parent_node):
        current_node = parent_node
        if not parent_node.child:
            parent_node.child.append(new_node)
            new_node.parent = parent_node
        else:
            current_node = parent_node.child[0]
            if (new_node.val[1] < current_node.val[0]): #on the left of current node    #<
                parent_node.child.insert(0, new_node)
                new_node.parent = parent_node
            elif(new_node.val[0] >= current_node.val[0]
                 and new_node.val[1] <= current_node.val[1]):   #child of current node
                self.add(new_node, current_node)
            else:
                for i in range(1, len(parent_node.child), 1):
                    if(new_node.val[0] >= parent_node.child[i].val[0]
                            and new_node.val[1] <= parent_node.child[i].val[1]):
                        self.add(new_node, parent_node.child[i])
                        return
                    elif(new_node.val[0] > parent_node.child[i-1].val[1]
                         and new_node.val[1] < parent_node.child[i].val[0]):    #>, <
                        parent_node.child.insert(i, new_node)
                        new_node.parent = parent_node
                        return
                if(new_node.val[0] > parent_node.child[-1].val[1]): #>
                    parent_node.child.append(new_node)
                    new_node.parent = parent_node
                else:
                    print("node can not insert.", new_node.val[0], new_node.val[1])

        return

    def delete(self, node):
        if not node.child:
            node.parent.child.remove(node)
        else:
            node_pos = node.parent.child.index(node)  #node postion at parent_child list
            for i in node.child:
                i.parent = node.parent
                node.parent.child.insert(node_pos, i)
                node_pos += 1
            node.parent.child.remove(node)
        return

    def pre_traversal(self, tree_root):  # traverse binary tree in pre-order
        if (tree_root != None):
            if tree_root != self.root:
                self.node_list.append(tree_root)
            for i in range(0, len(tree_root.child), 1):
                self.pre_traversal(tree_root.child[i])

    def acquire_list(self):
        self.pre_traversal(self.root)
        return self.node_list

    def find_parent(self, tree_node, bin):
        parent_start = tree_node.val[0]
        parent_end = tree_node.val[1]
        if(len(tree_node.child)!=0 and tree_node.val[0]<=bin and tree_node.val[1]>=bin):
            for i in tree_node.child:
                if(i.val[0]<=bin and i.val[1]>=bin):
                    parent_start, parent_end = self.find_parent(i, bin)
        return parent_start, parent_end

#visualization
def create_graph(G, node, pos={}, x=0, y=0, layer=1):
    key = str(node.val[0])+"-"+str(node.val[1])
    pos[key] = (x, y)
    child_num = len(node.child)
    if(child_num == 1):
        child = node.child[0]
        key_child = str(child.val[0]) + "-" + str(child.val[1])
        G.add_edge(key, key_child)
        (child_x, child_y) = (x, y - 1)
        create_graph(G, child, x=child_x, y=child_y, pos=pos, layer=layer+1)
    else:
        for i in range(0, child_num, 1):
            child = node.child[i]
            key_child = str(child.val[0]) + "-" + str(child.val[1])
            G.add_edge(key, key_child)
            (child_x, child_y) = (x - 1/2**layer + 2*1/2**layer/(child_num-1)*i, y - 1)
            create_graph(G, child, x=child_x, y=child_y, pos=pos, layer=layer+1)
    return (G, pos)

def draw(node, dirnames):
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx(graph, pos, ax=ax, node_size=400, fontsize=16)
    plt.savefig(os.path.join(dirnames, "multichild_tree.pdf"), format='pdf')

def scatter_plot(node_list, title=None, ylabel=None, p_parent=None, output=False):
    """
    :param node_list: list
    :param title: string
    :param ylabel: string
    :param p_parent: 'diff' means difference between node and its parent;
                    'quo' means quotient of node and its parent;
                    default means inherent info.
    :return:
    """
    all_boundary = []
    X = range(0, len(node_list))
    Y = []
    Color = []
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if(title != None):
        ax1.set_title(title)
    plt.xlabel('Node list with pre-order')
    if(ylabel != None):
        plt.ylabel(ylabel)
    for i in node_list:
        if(p_parent==None):
            Y.append(i.info)
        else:
            if(i.parent==None):
                Y.append(i.info)
            else:
                if(p_parent =='diff'):
                    Y.append(i.info - i.parent.info)
                    if(output==True and (i.info-i.parent.info > node_list[0].info)):
                        all_boundary.append([i.val[0], i.val[1]])   #start and end pos of true node
                elif(p_parent == 'quo'):
                    Y.append(i.info/i.parent.info)
        if(i.indicator == 'T'):
            Color.append('r')
        else:
            Color.append('g')
    ax1.scatter(X, Y, c=Color, marker='o')
    for i in range(0, len(node_list)):
        label = str(node_list[i].val[0]) + '-'+ str(node_list[i].val[1])
        # ax1.annotate(label, (i+1, node_list[i].info))
        ax1.text(i+1, Y[i], label, horizontalalignment='left', size='medium', color=Color[i])
    plt.show()
    if(output==True):
        return all_boundary







