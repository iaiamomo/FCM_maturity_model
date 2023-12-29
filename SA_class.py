import networkx as nx
from FCM_class import FCM

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

        cycles = {}
        pos_cycles = {}
        neg_cycles = {}
        for i in range(n_graphs):
            G : nx.DiGraph = self.model[i]
            cycl = list(nx.simple_cycles(G))
            cycles[i] = cycl
            pos_cycles[i] = 0
            neg_cycles[i] = 0
            for cycle in cycl:
                product = 1.0
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]  # To loop back to the first node
                    edge_weight = G[u][v]['weight']
                    product *= edge_weight
                
                if product > 0:
                    pos_cycles[i] += 1
                elif product < 0:
                    pos_cycles[i] += 1

        return cycles, pos_cycles, neg_cycles
    

def print_dict(d):
    for i in d:
        print(f"\tFCM {i}: {d[i]}")

def print_cycles(cycles, pos_cycles, neg_cycles):
    for i in cycles:
        print(f"\tFCM {i}: {len(cycles[i])}, {pos_cycles[i]} positive, {neg_cycles[i]} negative")
        

if __name__ == "__main__":
    n_fcm = 6
    model_type = 4
    c = 1
    iterations = 25
    fcm_obj = FCM(n_fcm, iterations, model_type, c)

    sa = StaticAnalysis(fcm_obj.model, fcm_obj.desc_graphs)

    n_graphs, depth, layer, density, n_nodes = sa.get_complexity()
    print(f"Number of graphs: {n_graphs}")
    print(f"Depth: {depth}")
    print(f"FCM Layers:")
    print_dict(layer)
    print(f"Density - how many connections exists wrt the maximum possible number of connections")
    print_dict(density)
    print(f"Number of nodes per FCM:")
    print_dict(n_nodes)

    degrees, values = sa.get_strenght()
    print(f"Nodes degree - number of incoming/outgoing edges per node per FCM")
    print_dict(degrees)
    print(f"Nodes values - weight of incoming/outgoing edges per node per FCM")
    print_dict(values)

    cycles, pos_cycles, neg_cycles = sa.get_tendency()
    print(f"Number of cycles per FCM:")
    print_cycles(cycles, pos_cycles, neg_cycles)