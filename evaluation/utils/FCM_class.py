import networkx as nx
import numpy as np
import json
import utils.FLT_class as FLT_class
import pandas as pd


model_path = "../model"
cases_path = "../cases"


class FCM:

    def __init__(self, n_fcm, iterations, lambdas, company_type, flt, new_values=[]):
        self.n_fcm = n_fcm
        self.iterations = iterations
        self.al_flt = flt
        self.model : list[nx.DiGraph] = self.build_model(company_type, new_values)
        self.lambdas = lambdas


    def build_model(self, company_type, new_values=[]):
        "Build the FCM model"
        graph_list = []
        desc_nodes_list = []
        self.desc_graphs = []
        idx_new_values = 0
        for i in range(self.n_fcm):
            idx = i+1
            # get weights and activation levels from csv files 
            ww = np.genfromtxt(f'{model_path}/{idx}_wm.csv', delimiter=',')
            al = pd.read_csv(f'{cases_path}/{company_type}/{idx}_al.csv', header=None).values

            # if doing genetic algorithm, change the activation levels
            if len(new_values) > 0:
                # modify activation levels based on genes, change only the technologies (not the main concept)
                al[0][0] = 0
                for x in range(1, len(al)):    # x is the row index
                    al[x][0] = new_values[idx_new_values]
                    idx_new_values+=1

            # create graph from weights and activation levels
            h = FCM.fcm_from_matrix_to_graph(ww, al, self.iterations+1, self.al_flt, new_values)

            # get description of the graph
            desc = json.load(open(f'{model_path}/{idx}_desc.json'))
            desc_nodes = desc['nodes']
            desc_nodes_list.append(desc_nodes)
            self.desc_graphs.append(desc)

            graph_list.append(h)

        return graph_list


    @staticmethod
    def fcm_from_matrix_to_graph(ww, al, iterations, flt : FLT_class.Fuzzy_Linguistic_Terms, new_values=[]):
        "Create a graph based on a given matrix"
        G = nx.DiGraph(depth=0)
        n = ww.shape[0]

        # nodes
        # "value" is an array representing the activation level through the iterations
        # "link" is the index of the graph to which the node is linked to
        if len(new_values) == 0:    # if doing inference
            for k in range(n):
                G.add_node(k, attr_dict = {"value":[0]*iterations, "link":al[k][1]})
                l_al = al[k][0]
                v_al = flt.get_value(l_al)
                G.nodes[k]['attr_dict']['value'][0] = round(v_al, 5)
        else:   # if doing genetic algorithm
            for k in range(n):
                G.add_node(k, attr_dict = {"value":[0]*iterations, "link":al[k][1]})
                G.nodes[k]['attr_dict']['value'][0] = round(al[k][0], 5)

        # edges
        for i in range(n):
            for j in range(n):
                if (ww[i][j] != 0): G.add_edge(i, j, weight = round(ww[i][j], 5))

        return G


    def run_fcm(self, threshold=0.001):
        self.model_out = []
        self.final_al = []
        "Run the FCM algorithm"
        for i in range(len(self.model)):
            lambda_value = self.lambdas[i+1]
            G, t = FCM.papageorgiou_alg_graph(self.model[i], start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
            for n in range(len(G.nodes)):
                G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
            self.model_out.append(G)
        max_iter = 0
        for i in range(len(self.model_out)):
            if len(self.model_out[i].nodes[0]['attr_dict']['value']) > max_iter:
                max_iter = len(self.model_out[i].nodes[0]['attr_dict']['value'])
        for i in range(len(self.model_out)):
            if max_iter > len(self.model_out[i].nodes[0]['attr_dict']['value']):
                for n in range(len(self.model_out[i].nodes)):
                    idx_iter = len(self.model_out[i].nodes[n]['attr_dict']['value'])
                    while idx_iter < max_iter:
                        self.model_out[i].nodes[n]['attr_dict']['value'].append(self.model_out[i].nodes[n]['attr_dict']['value'][-1])
                        idx_iter += 1
            # extract the final activation level from the final graph
            self.final_al.append(self.model_out[i].nodes[0]['attr_dict']['value'][-1])
        self.main_final_al = np.mean(self.final_al)
        return


    @staticmethod
    def papageorgiou_alg_graph(graph, start_iter, end_iter, lambda_value, threshold=0.001):
        "E.I. Papageorgiou, 'A new methodology for Decisions in Medical Informatics using fuzzy cognitive maps based on fuzzy rule-extraction techniques', Applied Soft Computing, vol. 11, Issue 1, p.p. 500-513, 2011."
        G = graph

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph
                # contribution of the incoming edges
                b = 0
                for edge in G.in_edges(node):
                    other, _ = edge
                    w_edge = G[other][node]['weight']
                    other_attr = G.nodes[other]['attr_dict']['value']
                    
                    c = 2 * other_attr[t-1] - 1  # C_j^t = 2 * A_j^(t-1) - 1

                    b += w_edge * c   # B_i^t = sum(w_ij * C_j^t)

                x = b + 2 * G.nodes[node]['attr_dict']['value'][t-1] - 1  # X_i^t = B_i^t + 2 * A_i^(t-1) - 1

                final_al = round(FCM.sigmoid(x, lambda_value), 5)  # A_i^t = sigmoid(X_i^t)

                G.nodes[node]['attr_dict']['value'][t] = final_al

                if node == 0:
                    if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                        return G, t

        return G, t


    @staticmethod
    def sigmoid(x, lambda_value):
        "Sigmoid function"
        return 1/(1+np.exp(-lambda_value*x))
