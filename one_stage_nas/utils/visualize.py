import math
import copy
from graphviz import Digraph

def model_visualize(model, save_dir, tie=True, return_cell=False):
    geno_cell = copy.deepcopy(model).module.cpu().genotype()
    visualize(geno_cell, save_dir, tie=tie)

    if return_cell:
        return geno_cell

def visualize(geno_cell, save_dir, tie=True):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])

    # cell  tie==True
    if tie == True:
        g.node("Pre_pre_cell", fillcolor='darkseagreen2')
        g.node("Pre_cell", fillcolor='darkseagreen2')

        node_num = len(geno_cell)//2
        for i in range(node_num):
            g.node(name='Node {}'.format(i), fillcolor='lightblue')

        for i in range(node_num):
            for k in [2*i, 2*i+1]:
                op, j = geno_cell[k]
                if op != 'none':
                    if j==1:
                        u = "Pre_pre_cell"
                        v = 'Node {}'.format(i)
                        g.edge(u, v, label=op, fillcolor='red')
                    elif j== 0:
                        u = "Pre_cell"
                        v = 'Node {}'.format(i)
                        g.edge(u, v, label=op, fillcolor='red')
                    else:
                        u = 'Node {}'.format(j-2)
                        v = 'Node {}'.format(i)
                        g.edge(u, v, label=op, fillcolor='gray')

        g.node('Cur_cell', fillcolor='palegoldenrod')
        for i in range(node_num):
            g.edge('Node {}'.format(i), 'Cur_cell', fillcolor='palegoldenrod')

    # cell tie == False
    else:
        for cell_id in range(len(geno_cell)):
            geno_cell_i = geno_cell[cell_id]

            if cell_id == 0:
                pre_pre_cell = 'stem1'
                pre_cell = 'stem2'
            elif cell_id == 1:
                pre_pre_cell = 'stem2'
                pre_cell = 'cell_0'
            elif cell_id > 1:
                pre_pre_cell = 'cell_{}'.format(cell_id-2)
                pre_cell = 'cell_{}'.format(cell_id-1)

            cur_cell = 'cell_{}'.format(cell_id)

            g.node(pre_pre_cell, fillcolor='darkseagreen2')
            g.node(pre_cell, fillcolor='darkseagreen2')

            node_num = len(geno_cell_i) // 2
            for i in range(node_num):
                g.node(name='C{}_N{}'.format(cell_id, i), fillcolor='lightblue')

            for i in range(node_num):
                for k in [2 * i, 2 * i + 1]:
                    op, j = geno_cell_i[k]
                    if op != 'none':
                        if j == 1:
                            u = pre_pre_cell
                            v = 'C{}_N{}'.format(cell_id, i)
                            g.edge(u, v, label=op, fillcolor='red')
                        elif j == 0:
                            u = pre_cell
                            v = 'C{}_N{}'.format(cell_id, i)
                            g.edge(u, v, label=op, fillcolor='red')
                        else:
                            u = 'C{}_N{}'.format(cell_id, j - 2)
                            v = 'C{}_N{}'.format(cell_id, i)
                            g.edge(u, v, label=op, fillcolor='gray')

            g.node(cur_cell, fillcolor='palegoldenrod')
            for i in range(node_num):
                g.edge('C{}_N{}'.format(cell_id, i), cur_cell, fillcolor='palegoldenrod')

    g.render(save_dir, view=False)




