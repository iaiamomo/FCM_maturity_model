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
'''

class StaticAnalysis:

    def __init__(self, model, description):
        self.model : list[nx.DiGraph] = model
        self.description : dict = description


    def get_complexity(self):
        self.n_graphs = len(self.model)
        
        self.layer = {}    # layer = FCM level
        self.depth = 0
        for i in range(self.n_graphs):
            self.layer[i] = self.description[i]['layer'] 
            if self.layer[i] > self.depth:
                self.depth = self.layer[i]

        self.density = {}    # density = n_edges / n_nodes
        for i in range(self.n_graphs):
            self.density[i] = round(nx.density(self.model[i]), 2)

        self.n_nodes = {}    # n_nodes = number of nodes
        for i in range(self.n_graphs):
            self.n_nodes[i] = len(self.model[i].nodes)

        return self.n_graphs, self.depth, self.layer, self.density, self.n_nodes


    def get_strenght(self):
        n_graphs = len(self.model)

        self.degrees = {}
        self.values = {}
        for i in range(n_graphs):
            G : nx.DiGraph = self.model[i]
            self.degrees[i] = G.degree()
            self.values[i] = G.degree(weight='weight')
            new_values = []
            for elem in self.values[i]:
                elem = list(elem)
                new_values.append((elem[0], round(elem[1], 2)))
            self.values[i] = new_values

        return self.degrees, self.values
    
    def find_most_important_nodes(self):
        self.imp_degree = {}
        self.imp_value = {}

        print(len(self.description))

        for i in range(self.n_graphs):
            sorted_degree = sorted(self.degrees[i], key=lambda x: x[1], reverse=True)
            idx_deg = sorted_degree[0][0]
            value_deg = sorted_degree[0][1]
            if idx_deg == 0:
                idx_deg = sorted_degree[1][0]
                value_deg = sorted_degree[1][1]
            idx_deg += 1
            name_idx_deg = self.description[i]['nodes'][str(idx_deg)]
            self.imp_degree[i] = (idx_deg, name_idx_deg, value_deg)

            sorted_val = sorted(self.values[i], key=lambda x: x[1], reverse=True)
            idx_val = sorted_val[0][0]
            value_val = sorted_val[0][1]
            if idx_val == 0:
                idx_val = sorted_val[1][0]
                value_val = sorted_val[1][1]
            idx_val += 1
            name_idx_val = self.description[i]['nodes'][str(idx_val)]
            self.imp_value[i] = (idx_val, name_idx_val, value_val)

        return self.imp_degree, self.imp_value

    
    def print_dict(self, d):
        for i in d:
            print(f"\tFCM {i} - {self.description[i]['main']}: {d[i]}")
        

if __name__ == "__main__":
    n_fcm = 6
    model_type = 5
    c = 4
    iterations = 25
    fcm_obj = FCM(n_fcm, iterations, model_type, c)

    sa = StaticAnalysis(fcm_obj.model, fcm_obj.desc_graphs)

    sa.get_complexity()
    print(f"Number of graphs: {sa.n_graphs}")
    print(f"Depth: {sa.depth}")
    print(f"FCM Layers:")
    sa.print_dict(sa.layer)
    print(f"Density - how many connections exists wrt the maximum possible number of connections")
    sa.print_dict(sa.density)
    print(f"Number of nodes per FCM:")
    sa.print_dict(sa.n_nodes)

    sa.get_strenght()
    print(f"Nodes degree - number of incoming/outgoing edges per node per FCM")
    sa.print_dict(sa.degrees)
    print(f"Nodes values - weight of incoming/outgoing edges per node per FCM")
    sa.print_dict(sa.values)

    sa.find_most_important_nodes()
    print(f"Most important nodes per FCM (degree)")
    sa.print_dict(sa.imp_degree)
    print(f"Most important nodes per FCM (value)")
    sa.print_dict(sa.imp_value)