import networkx as nx
from main import create_model

'''
Complexity
    density (number of nodes and interactions), 
    depth (number of layers), and 
    breadth (sub-FCMs and their nodes)
Strength
    weight and number of incoming edges (in-value, in-degree)
    weight and number of outgoing edges (out-value, out-degree)
Tendency
    number of feedback cycles in the graph (positive and negative)
    number of cycles in which each node takes part
'''

class StaticAnalysis:

    def __init__(self, model, description):
        self.model : list[nx.DiGraph] = model
        self.description : dict = description


    def get_complexity(self):
        n_graphs = len(self.model)

        layer = {}    # layer = FCM level
        depth = 0
        for i in range(n_graphs):
            layer[i] = self.description[i]['layer'] 
            if layer[i] > depth:
                depth = layer[i]

        density = {}    # density = n_edges / n_nodes
        for i in range(n_graphs):
            density[i] = round(nx.density(self.model[i]), 2)

        n_nodes = {}    # n_nodes = number of nodes
        for i in range(n_graphs):
            n_nodes[i] = len(self.model[i].nodes)

        return n_graphs, depth, layer, density, n_nodes


    def get_strenght(self):
        n_graphs = len(self.model)

        degrees = {}
        values = {}
        for i in range(n_graphs):
            G : nx.DiGraph = self.model[i]
            degrees[i] = G.degree()
            values[i] = G.degree(weight='weight')
            new_values = []
            for elem in values[i]:
                elem = list(elem)
                new_values.append((elem[0], round(elem[1], 2)))
            values[i] = new_values

        return degrees, values
    

    def get_tendency(self):
        n_graphs = len(self.model)

        for i in range(n_graphs):
            G : nx.DiGraph = self.model[i]
            cycles = nx.simple_cycles(G)
            print(list(cycles))
    

def print_dict(d):
    for i in d:
        print(f"\t{i}: {d[i]}")
        

if __name__ == "__main__":
    n_fcm = 6
    model_type = 4
    c = 1
    iterations = 25
    model, desc_nodes_list, desc_graphs = create_model(n_fcm, model_type=model_type, comp_type=c, iterations=iterations)

    sa = StaticAnalysis(model, desc_graphs)

    n_graphs, depth, layer, density, n_nodes = sa.get_complexity()
    print(f"Number of graphs: {n_graphs}")
    print(f"Depth: {depth}")
    print(f"Layer:")
    print_dict(layer)
    print(f"Density:")
    print_dict(density)
    print(f"Number of nodes:")
    print_dict(n_nodes)

    degrees, values = sa.get_strenght()
    print(f"Nodes degree:")
    print_dict(degrees)
    print(f"Nodes values:")
    print_dict(values)
