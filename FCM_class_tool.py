import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from random import randint
import copy
import FLT_class
import pandas as pd
import glob
import random

model_path = "model"
cases_path = "cases"
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

class FCM:

    def __init__(self, iterations, structure, activation_level, flt, new_values=[]):
        self.iterations = iterations
        self.al_flt = flt
        self.lambdas = {}
        self.model : dict[nx.DiGraph] = self.build_model(structure, activation_level, new_values=new_values)


    def build_model(self, structure, activation_level, new_values=[]):
        "Build the FCM model"
        graph_list = {}
        self.desc_graphs = {}
        idx_new_values = 0

        self.idx_to_remove = []

        self.root = None
        intermediate = []
        final = []
        
        nodes = structure['nodes']
        transitions = structure['transitions']

        self.all_nodes = {}
        for node in nodes:
            '''
            {
                "id": 0,
                "role": "root",
                "meanings": ["Smart Manufacturing"],
                "targets": [1, 7, 8, 10, 11, 12]
            }
            '''
            node_id = node['id']
            node_role = node['role']
            if node_role == "root":
                self.root = node_id
                self.all_nodes[node_id] = node
                continue
            elif node_role == "intermediate":
                intermediate.append(node_id)
                self.all_nodes[node_id] = node
                self.lambdas[node_id] = node['lambda']
            elif node_role == "final":
                final.append(node_id)
                self.all_nodes[node_id] = node
        
        transitions_from = {}
        for elem in transitions:
            '''
            {
                "from": 0,
                "to": 1,
                "weight": 0.5
            }
            '''
            from_node = elem['from']
            to_node = elem['to']
            weight = elem['weight']
            if from_node not in transitions_from:
                transitions_from[from_node] = {}
            transitions_from[from_node][to_node] = weight

            if to_node in intermediate:
                if 'destination_of' not in self.all_nodes[to_node]:
                    self.all_nodes[to_node]['destination_of'] = []
                self.all_nodes[to_node]['destination_of'].append(from_node)

        all_nodes_al = {}
        for node in activation_level:
            # { "id": 1, "label": "CAD, CAM, PLM", "weight": 0.0, "enabled": true }
            node_id = node['id']
            all_nodes_al[node_id] = node
            if node_id in intermediate:
                node_enabled = node['enabled']
                if not node_enabled:
                    self.idx_to_remove.append(node_id)

        for node_id in intermediate:
            if node_id not in self.idx_to_remove:
                node_info = self.all_nodes[node_id]

                G = nx.DiGraph(depth=0)
                destination_of = node_info["destination_of"]

                # add nodes
                # "value" is an array representing the activation level through the iterations
                if len(new_values) == 0:    # if doing inference
                    # add the intermediate node
                    G.add_node(node_id, attr_dict = {"value":[0]*iterations})
                    l_al = all_nodes_al[node_id]['weight']
                    v_al = flt.get_value(l_al)
                    G.nodes[node_id]['attr_dict']['value'][0] = round(v_al, 5)
                    # add final nodes
                    for other_node_id in destination_of:
                        G.add_node(other_node_id, attr_dict = {"value":[0]*iterations})
                        l_al = all_nodes_al[other_node_id]['weight']
                        v_al = flt.get_value(l_al)
                        G.nodes[other_node_id]['attr_dict']['value'][0] = round(v_al, 5)
                else:   # if doing genetic algorithm
                    # add the intermediate node
                    G.add_node(node_id, attr_dict = {"value":[0]*iterations})
                    l_al = all_nodes_al[node_id]['weight']
                    v_al = flt.get_value(l_al)
                    G.nodes[node_id]['attr_dict']['value'][0] = round(v_al, 5)
                    # add final nodes
                    for other_node_id in destination_of:
                        G.add_node(other_node_id, attr_dict = {"value":[0]*iterations})
                        l_al = all_nodes_al[other_node_id]['weight']
                        G.nodes[other_node_id]['attr_dict']['value'][0] = round(l_al, 5)

                nodes_to_consider = [node_id] + destination_of
                # edges of the graph
                for i in nodes_to_consider:
                    for j in nodes_to_consider:
                        ww_ij = transitions_from[i][j] if i in transitions_from and j in transitions_from[i] else 0
                        if (ww_ij != 0): G.add_edge(i, j, weight = round(ww_ij, 5))

                graph_list[node_id] = G

        return graph_list
    
    def run_fcm(self, threshold=0.001):
        "Run the FCM algorithm"
        self.model_out = {}

        # run the algorithm for each FCM
        for key in self.model.keys():
            lambda_value = self.lambdas[key]
            G, t = FCM.papageorgiou_alg_graph(self.model[key], key, start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
            for n in G.nodes:
                G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
            self.model_out[key] = G

        # compute the number of max iterations in the graphs
        max_iter = 0
        for key in self.model_out.keys():
            if len(self.model_out[key].nodes[key]['attr_dict']['value']) > max_iter:
                max_iter = len(self.model_out[key].nodes[key]['attr_dict']['value'])
        
        # add the missing iterations to the activation levels of the nodes
        # the last value is repeated until the max_iter is reached
        for key in self.model_out.keys():
            if max_iter > len(self.model_out[key].nodes[key]['attr_dict']['value']):
                for n in self.model_out[key].nodes:
                    idx_iter = len(self.model_out[key].nodes[n]['attr_dict']['value'])
                    while idx_iter < max_iter:
                        self.model_out[key].nodes[n]['attr_dict']['value'].append(self.model_out[key].nodes[n]['attr_dict']['value'][-1])
                        idx_iter += 1

        # compute the final activation level of the main node
        weight_mean = 0
        n_elem = 0
        for key in self.model_out.keys():
            weight_mean += self.model_out[key].nodes[key]['attr_dict']['value'][-1]
            n_elem += 1
        self.main_final_al = weight_mean / n_elem

        return


    @staticmethod
    def papageorgiou_alg_graph(graph, key_node, start_iter, end_iter, lambda_value, threshold=0.001):
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
                    
                    # C_j^t = 2 * A_j^(t-1) - 1
                    c = 2 * other_attr[t-1] - 1

                    # B_i^t = sum(w_ij * C_j^t)
                    b += w_edge * c

                # X_i^t = B_i^t + 2 * A_i^(t-1) - 1
                x = b + 2 * G.nodes[node]['attr_dict']['value'][t-1] - 1

                # A_i^t = sigmoid(X_i^t)
                final_al = round(FCM.sigmoid(x, lambda_value), 5)

                G.nodes[node]['attr_dict']['value'][t] = final_al

                if node == key_node:
                    if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                        return G, t

        return G, t


    @staticmethod
    def sigmoid(x, lambda_value):
        "Sigmoid function"
        return 1/(1+np.exp(-lambda_value*x))
    

    def generate_al_values(self):
        final_al_nodes = []
        for key in self.model_out.keys():
            for n in self.model_out[key].nodes:
                final_al = self.model_out[key].nodes[n]['attr_dict']['value'][-1]
                final_flt = self.al_flt.get_linguisitic_term(final_al)
                dict_al = {
                    "id": n, 
                    "label": ", ".join(self.all_nodes[n]['meanings']),
                    "weight": final_flt,
                    "numeric_weight": final_al
                }
                final_al_nodes.append(dict_al)
        final_main_al = self.al_flt.get_linguisitic_term(self.main_final_al)
        main_al = {
            "id": self.root,
            "label": ", ".join(self.all_nodes[self.root]['meanings']), 
            "weight": final_main_al,
            "numeric_weight": self.main_final_al
        }
        final_al_nodes.append(main_al)
        return final_al_nodes                


    def print_weights_nodes(self):
        "Print the weights of the nodes of the final graph"
        for key in self.model_out.keys():
            # grafo i-esimo
            to_print = f"FCM {', '.join(self.all_nodes[key]['meanings'])}\n"
            for n in self.model_out[key].nodes:
                # nodo n-esimo
                to_print += f"\t{', '.join(self.all_nodes[n]['meanings'])}:\t"
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
            for j in self.model_out[key].nodes:
                results_in.append(self.model_out[key].nodes[j]['attr_dict']['value'][0])
                results_out.append(self.model_out[key].nodes[j]['attr_dict']['value'][-1])
            initial_activation_levels.append(results_in)
            final_activation_level.append(results_out)
            name_fcm.append(", ".join(self.all_nodes[key]['meanings']))
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
        y_val = g.nodes[key]['attr_dict']['value']
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
            y_val_mean.append(g.nodes[key]['attr_dict']['value'][idx_iter])
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

    return


if __name__ == "__main__":
    config = json.load(open("config.json"))
    to_remove = config['to_remove']
    company = config['case']

    flt = FLT_class.define_al_fuzzy()
    flt.plot_triangle()

    flt2 = FLT_class.define_wm_fuzzy()
    flt2.plot_triangle()

    iterations = 100  # number of iterations
    threshold = 0.001
    print_status = False    # print status of the sub-fcms

    # load graph structure and activation levels
    structure = json.load(open(f'single_file.json'))
    activation_level = json.load(open(f'current_al.json')) 

    print(f"Algorithm: Papageorgiou, Iterations: {iterations}, Company Type: {company}")
    fcm_obj = FCM(iterations, structure, activation_level, flt)
    plot_sigmoid(fcm_obj.lambdas)
    fcm_obj.run_fcm(threshold)

    # from the model output extract the activation levels to pass to the tool
    json_output = fcm_obj.generate_al_values()
    with open(f'final_al.json', 'w') as outfile:
        json.dump(json_output, outfile, indent=4)

    fcm_obj.print_results(flt)
    linguistic_al = flt.get_linguisitic_term(fcm_obj.main_final_al)
    print(f"Final activation level: {fcm_obj.main_final_al} ({linguistic_al})")
    if print_status:
        fcm_obj.print_weights_nodes()
    print("\n")

    plot_al_values_graphs(fcm_obj, company, colors)

    plt.show()
