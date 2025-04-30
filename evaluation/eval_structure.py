import networkx as nx
from utils.FCM_class import FCM
from utils.FLT_class import * 

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

        self.n_edges = {}    # n_edges = number of edges
        self.n_nodes = {}    # n_nodes = number of nodes
        self.density = {}    # density = n_edges / (n_nodes * (n_nodes - 1))
        for i in range(self.n_graphs):
            self.density[i] = round(nx.density(self.model[i]), 2)
            self.n_edges[i] = len(self.model[i].edges)
            self.n_nodes[i] = len(self.model[i].nodes)

        return self.n_graphs, self.depth, self.layer, self.density, self.n_nodes


    def get_strenght(self):
        n_graphs = len(self.model)

        self.degrees = {}
        self.values = {}
        self.centrality = {}
        for i in range(n_graphs):
            G : nx.DiGraph = self.model[i]
            self.degrees[i] = G.out_degree()
            self.values[i] = G.out_degree(weight='weight')
            new_values = []
            for elem in self.values[i]:
                elem = list(elem)
                new_values.append((elem[0], round(elem[1], 2)))
            self.values[i] = new_values
            self.centrality[i] = nx.degree_centrality(G)
            new_centrality = []
            for elem in self.centrality[i]:
                new_centrality.append((elem, round(self.centrality[i][elem], 2)))
            self.centrality[i] = new_centrality

        return self.degrees, self.values, self.centrality
    
    def find_most_important_nodes(self):
        self.imp_degree = {}
        self.imp_value = {}

        for i in range(self.n_graphs):
            sorted_degree = sorted(self.degrees[i], key=lambda x: x[1], reverse=True)
            res = []
            idx_deg = sorted_degree[0][0]
            value_deg = sorted_degree[0][1]
            name_idx_deg = self.description[i]['nodes'][str(idx_deg+1)]
            res.append((idx_deg, name_idx_deg, value_deg))
            for j in range(len(sorted_degree)):
                if j == idx_deg:
                    continue
                if sorted_degree[j][1] == value_deg:
                    idx_deg = sorted_degree[j][0]
                    value_deg = sorted_degree[j][1]
                    name_idx_deg = self.description[i]['nodes'][str(idx_deg+1)]
                    res.append((idx_deg, name_idx_deg, value_deg))
            self.imp_degree[i] = res

            sorted_val = sorted(self.values[i], key=lambda x: x[1], reverse=True)
            res = []
            idx_val = sorted_val[0][0]
            value_val = sorted_val[0][1]
            name_idx_val = self.description[i]['nodes'][str(idx_val+1)]
            res.append((idx_val, name_idx_val, value_val))
            for j in range(len(sorted_val)):
                if j == idx_val:
                    continue
                if sorted_val[j][1] == value_val:
                    idx_val = sorted_val[j][0]
                    value_val = sorted_val[j][1]
                    name_idx_val = self.description[i]['nodes'][str(idx_val+1)]
                    res.append((idx_val, name_idx_val, value_val))            
            self.imp_value[i] = res

        return self.imp_degree, self.imp_value

    
    def print_dict(self, d):
        for i in d:
            print(f"\tFCM {i+1} - {self.description[i]['main']}: {d[i]}")
        

if __name__ == "__main__":
    c = "low"
    
    lambdas = {
        1: 0.83,
        2: 0.85,
        3: 0.81,
        4: 0.91,
        5: 0.735
    }

    n_fcm = 5
    iterations = 100
    flt = define_wm_fuzzy()
    fcm_obj = FCM(n_fcm, iterations, lambdas, c, flt)

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
    print(f"Number of edges per FCM:")
    sa.print_dict(sa.n_edges)

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