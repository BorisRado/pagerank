import networkx as nx
import sys

def get_nodes(line):
    return list(map(int, line.split()))

def read_graph(filename):
    G = nx.DiGraph()
    
    with open(filename, 'r') as file:
        nodes_count, _ = get_nodes(file.readline())
        G.add_nodes_from(range(nodes_count))

        for line in file:
            from_node, to_node = get_nodes(line)
            G.add_edge(from_node, to_node)

    return G

def compute_pagerank(filename):
    G = read_graph(filename)
    pagerank = nx.pagerank(G)
    return pagerank

def compare_pageranks(true_pagerank, filename, eps):
    with open(filename, 'r') as file:
        for idx, line in enumerate(file):
            value = float(line)
            assert abs(value - true_pagerank[idx]) < eps, \
                f'Too large difference detected! {value} {true_pagerank[idx]} at line {idx}'
    print('Pageranks compared. No difference detected.')

if __name__ == '__main__':

    assert len(sys.argv) == 3, 'Need to provide the input graph and the file,'\
        'to which to match the results'
    pagerank = compute_pagerank(sys.argv[1])
    
    eps = 1e-3 # maximum tolerance for the difference
    compare_pageranks(pagerank, sys.argv[2], eps)