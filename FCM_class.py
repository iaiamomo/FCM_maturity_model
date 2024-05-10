import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from random import randint
import copy
import FLT_class
import pandas as pd
import glob

model_path = "model"
cases_path = "cases"
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

class FCM:

    def __init__(self, n_fcm, iterations, lambdas, company_type, flt, new_values=[], to_remove=[]):
        self.n_fcm = n_fcm
        self.iterations = iterations
        self.al_flt = flt
        self.lambdas = lambdas
        self.model : dict[nx.DiGraph] = self.build_model(company_type, new_values=new_values, to_remove=to_remove)


    def build_model(self, company_type, new_values=[], to_remove=[]):
        "Build the FCM model"
        graph_list = {}
        self.desc_graphs = {}
        self.weights_nodes = {}
        idx_new_values = 0

        self.idx_to_remove = []
        desc_files = glob.glob(f"{model_path}/*.json")
        for i in range(len(desc_files)):
            file_to_open = desc_files[i]
            data = json.load(open(file_to_open))
            if data['main'] in to_remove:
                self.idx_to_remove.append(i)
                self.lambdas.pop(i)

        for idx in range(1, self.n_fcm+1):
            if idx in self.idx_to_remove:
                continue

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
            self.desc_graphs[idx] = desc
            self.weights_nodes[idx] = desc['weight']

            graph_list[idx] = h

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
        self.model_out = {}
        "Run the FCM algorithm"
        for key in self.model.keys():
            lambda_value = self.lambdas[key]
            G, t = FCM.papageorgiou_alg_graph(self.model[key], start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
            for n in range(len(G.nodes)):
                G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
            self.model_out[key] = G
        max_iter = 0
        for key in self.model_out.keys():
            if len(self.model_out[key].nodes[0]['attr_dict']['value']) > max_iter:
                max_iter = len(self.model_out[key].nodes[0]['attr_dict']['value'])
        for key in self.model_out.keys():
            if max_iter > len(self.model_out[key].nodes[0]['attr_dict']['value']):
                for n in range(len(self.model_out[key].nodes)):
                    idx_iter = len(self.model_out[key].nodes[n]['attr_dict']['value'])
                    while idx_iter < max_iter:
                        self.model_out[key].nodes[n]['attr_dict']['value'].append(self.model_out[key].nodes[n]['attr_dict']['value'][-1])
                        idx_iter += 1

        weight_mean = 0
        n_elem = 0
        for key in self.model_out.keys():
            weight_mean += self.weights_nodes[key] * self.model_out[key].nodes[0]['attr_dict']['value'][-1]
            n_elem += 1
        self.main_final_al = weight_mean / n_elem

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


    def print_weights_nodes(self):
        "Print the weights of the nodes of the final graph"
        for key in self.model_out.keys():
            # grafo i-esimo
            to_print = f"FCM {self.desc_graphs[key]['main']}\n"
            for n in range(len(self.model_out[key].nodes)):
                # nodo n-esimo
                to_print += f"\t{self.desc_graphs[key]['nodes'][str(n+1)]}:\t"
                to_print += f"{self.model_out[key].nodes[n]['attr_dict']['value']}\n"
                pass
            print(to_print)


    def extract_weights(self):
        "Extract the weights from the final graph"
        initial_activation_levels = []
        final_activation_level = []
        name_fcm = []
        for key in self.model_out.keys():
            results_in = []
            results_out = []
            for j in range(len(self.model_out[key].nodes)):
                results_in.append(self.model_out[key].nodes[j]['attr_dict']['value'][0])
                results_out.append(self.model_out[key].nodes[j]['attr_dict']['value'][-1])
            initial_activation_levels.append(results_in)
            final_activation_level.append(results_out)
            name_fcm.append(self.desc_graphs[key]['main'])
        return initial_activation_levels, final_activation_level, name_fcm


    def print_results(self, flt):
        "Print the initial and final activation levels of the nodes in the final graph"
        print("Activation levels of nodes in the FCMs")
        initial_activation_levels, final_activation_level, name_fcm = self.extract_weights()

        ling_initial = copy.deepcopy(initial_activation_levels)
        ling_final = copy.deepcopy(final_activation_level)
        for i in range(len(initial_activation_levels)):
            for j in range(len(initial_activation_levels[i])):
                ling_initial[i][j] = flt.get_linguisitic_term(ling_initial[i][j])
                ling_final[i][j] = flt.get_linguisitic_term(ling_final[i][j])

        for i in range(len(initial_activation_levels)):
            print(f"FCM {name_fcm[i]}")
            print(f"\t Initial: {initial_activation_levels[i]}")
            print(f"\t\t  {ling_initial[i]}")
            print(f"\t Final: {final_activation_level[i]}")
            print(f"\t\t  {ling_final[i]}")


def plot_al_values_graphs(models, company, colors):
    "Plot the activation levels of the main node of multiple FCM"
    if "_" in company:
        company = " ".join(company.split("_"))
    plt.figure()
    plt.grid()
    n_iter = 100
    idx_color = 0
    for key in models.model_out.keys():
        g = models.model_out[key]
        y_val = g.nodes[0]['attr_dict']['value']
        if len(y_val) < n_iter:
            n_iter = len(y_val)
        x_val = list(range(len(y_val)))
        plt.plot(x_val, y_val, color=colors[idx_color], label=f"FCM {key}")
        idx_color += 1
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Activation Level")
    plt.title(f"Company {company}")

    idx_iter = 0
    y_val_main = []
    while idx_iter < n_iter:
        y_val_mean = []
        for key in models.model_out.keys():
            g = models.model_out[key]
            y_val_mean.append(g.nodes[0]['attr_dict']['value'][idx_iter])
        y_val_main.append(np.mean(y_val_mean))
        idx_iter += 1
    plt.figure()
    plt.plot(list(range(len(y_val_main))), y_val_main)
    plt.title(f"Main Activation Level - Company {company}")
    plt.ylim(0, 1)
    plt.xlabel("Iterations")
    plt.ylabel("Activation Level")
    plt.grid()

    plt.show(block=False)


def plot_sigmoid(lambda_values):
    plt.figure()

    c_i = 0
    x = np.linspace(-5, 5, 100)
    for i in lambda_values:
        lambda_value = lambda_values[i]
        y = []
        for j in x:
            y.append(FCM.sigmoid(j, lambda_value))
        plt.plot(x, y, color=colors[c_i], label=f'lambda = {lambda_value}')
        plt.axvline(x=-lambda_value, linestyle='--', color=colors[c_i])
        plt.axvline(x=lambda_value, linestyle='--', color=colors[c_i])
        c_i += 1

    plt.title(f'Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('Sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    return colors


if __name__ == "__main__":
    config = json.load(open("config.json"))
    to_remove = config['to_remove']
    company = config['case']

    flt = FLT_class.define_al_fuzzy()
    flt.plot_triangle()

    flt2 = FLT_class.define_wm_fuzzy()
    flt2.plot_triangle()

    n_fcm = 5   # number of sub-fcms
    iterations = 100  # number of iterations
    threshold = 0.001
    print_status = False    # print status of the sub-fcms

    #lambda determina quanto il modello è sensibile ai cambiamenti dei AL
    #lamda grande -> più sensibile ai cambiamenti, tende a 0 o 1
    #lamda piccolo -> meno sensibile ai cambiamenti, tende a 0.5
    lambdas = {
        1: 0.83,
        2: 0.85,
        3: 0.81,
        4: 0.91,
        5: 0.735
    }
    colors = plot_sigmoid(lambdas)

    print(f"Algorithm: Papageorgiou, Iterations: {iterations}, Company Type: {company}")
    fcm_obj = FCM(n_fcm, iterations, lambdas, company, flt, to_remove=to_remove)
    fcm_obj.run_fcm(threshold)
    fcm_obj.print_results(flt)
    linguistic_al = flt.get_linguisitic_term(fcm_obj.main_final_al)
    print(f"Final activation level: {fcm_obj.main_final_al} ({linguistic_al})")
    if print_status:
        fcm_obj.print_weights_nodes()
    print("\n")

    plot_al_values_graphs(fcm_obj, company, colors)

    plt.show()
