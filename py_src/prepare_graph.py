import sys

def iterate_edges(filename):
    processing_nodes = False
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('*'):
                processing_nodes = not processing_nodes
                continue
            if line.startswith('#') or processing_nodes:
                continue
            yield line

def write_corrected_graph(nodes, edges_count, filename):
    mapping = {int(node): idx for idx, node in enumerate(nodes)}
    tmp = filename.split('.')
    out_filename = f'{".".join(tmp[:-1])}_out.{tmp[-1]}'
    print(f'\tWrinting new graph to: {out_filename}')
    with open(f'{out_filename}', 'w') as out:
        out.write(f'{len(nodes)}\t{edges_count}\n')
        for line in iterate_edges(filename):
            if line.startswith('#'): continue
            pair = [mapping[int(v)] for v in line.split()]
            out.write(f'{pair[0]}\t{pair[1]}\n')

def run(filename):
    ''' 
    Reads the graph in filename, and prepares it for further processing. It moves
    all the nodes to the range [0:nodes_count-1], and prints the corrected graph
    to a file. Also, prints the nodes mapping to another file (in case that the)
    '''
    print(f'Processing file: {filename}')
    nodes = set()
    edges_count = 0
    for line in iterate_edges(filename):
        tmp = line.split()
        nodes.add(int(tmp[0]))
        nodes.add(int(tmp[1]))
        edges_count += 1
    print(f'\tNumber of nodes: {len(nodes)}')
    print(f'\tMax node: {max(nodes)}')
    write_corrected_graph(nodes, edges_count, filename)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Need to provide file name'
    run(sys.argv[1])