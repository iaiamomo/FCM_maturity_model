import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from random import randint
import copy
import FLT_class
import pandas as pd

class FCM:

    def __init__(self, n_fcm, iterations, company_type, flt, new_values=[]):
        self.n_fcm = n_fcm
        self.iterations = iterations
        self.al_flt = flt
        self.model : list[nx.DiGraph] = self.build_model(company_type, new_values)


    def build_model(self, company_type, new_values=[]):
        "Build the FCM model"
        graph_list = []
        desc_nodes_list = []
        self.desc_graphs = []
        idx_new_values = 0
        for i in range(self.n_fcm):
            # get weights and activation levels from csv files 
            ww = np.genfromtxt(f'model/{i}_wm.csv', delimiter=',')
            al = pd.read_csv(f'cases/{company_type}/{i}_al.csv', header=None).values

            # if doing genetic algorithm, change the activation levels
            if len(new_values) > 0:
                if i == 0:  # if main concept set to 0
                    for x in range(len(al)):
                        al[x][0] = 0
                else:   # if sub-concepts
                    # modify activation levels based on genes, change only the technologies (not the main concept)
                    al[0][0] = 0
                    for x in range(1, len(al)):    # x is the row index
                        al[x][0] = new_values[idx_new_values]
                        idx_new_values+=1

            # create graph from weights and activation levels
            h = FCM.fcm_from_matrix_to_graph(ww, al, 0, self.iterations+1, self.al_flt, new_values)

            # get description of the graph
            desc = json.load(open(f'model/{i}_desc.json'))
            desc_main = desc['main']
            desc_nodes = desc['nodes']
            desc_nodes_list.append(desc_nodes)
            self.desc_graphs.append(desc)

            graph_list.append(h)

        return graph_list


    @staticmethod
    def fcm_from_matrix_to_graph(ww, al, depth, iterations, flt : FLT_class.Fuzzy_Linguistic_Terms, new_values=[]):
        "Create a graph based on a given matrix"
        G = nx.DiGraph(depth=depth)
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
    
    
    @staticmethod
    def draw(graph, description):
        "Draw the graph"
        pos = nx.spring_layout(graph)
        colors = []
        for i in range(len(graph.nodes)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        nx.draw(graph, pos, with_labels=True, node_size=900, node_color=colors)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        legend_labels = []
        i = 0
        for _, desc in description['nodes'].items():
            legend_labels.append(plt.Circle((0, 0), 0.1, color=colors[i], label=f"{i}: {desc}"))
            i+=1
        plt.legend(handles=legend_labels, loc='upper left', title=description['main'])
    
        plt.show()


    def run_fcm(self, lambda_value, threshold=0.001):
        "Run the FCM algorithm"
        graph_list, t = FCM.papageorgiou_alg_graph(self.model[:], g_index=0, start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)

        # refine the final activation level array
        #print(f"FINISHED AT ITERATION {t}")
        for i in range (len(graph_list)):
            G = graph_list[i]
            for n in range(len(G.nodes)):
                G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
            graph_list[i] = G

        self.model_out = graph_list

        # extract the final activation level from the final graph
        n_main_concept = len(graph_list[0].nodes)-1
        self.final_activation_level = graph_list[0].nodes[n_main_concept]['attr_dict']['value'][-1]

        return


    @staticmethod
    def papageorgiou_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value, threshold=0.001):
        "E.I. Papageorgiou, 'A new methodology for Decisions in Medical Informatics using fuzzy cognitive maps based on fuzzy rule-extraction techniques', Applied Soft Computing, vol. 11, Issue 1, p.p. 500-513, 2011."
        G = graph_list[g_index]

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph
                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list, t = FCM.papageorgiou_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value, threshold)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

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

                if g_index == 0:
                    n_nodes = len(G.nodes)
                    if node == n_nodes-1:
                        if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                            #print(f"Threshold reached at iteration {t}")
                            #print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                            graph_list[g_index] = G
                            return graph_list, t

        graph_list[g_index] = G
        return graph_list, t


    @staticmethod
    def sigmoid(x, lambda_value):
        "Sigmoid function"
        return 1/(1+np.exp(-lambda_value*x))


    def print_weights_nodes(self):
        "Print the weights of the nodes of the final graph"
        for i in range(len(self.desc_graphs)):
            # grafo i-esimo
            to_print = f"FCM {i}\n"
            for n in range(len(self.model_out[i].nodes)):
                # nodo n-esimo
                to_print += f"\t{self.desc_graphs[i]['nodes'][str(n+1)]}:\t"
                to_print += f"{self.model_out[i].nodes[n]['attr_dict']['value']}\n"
                pass
            print(to_print)


    def extract_weights(self):
        "Extract the weights from the final graph"
        initial_activation_levels = []
        final_activation_level = []
        for i in range(len(self.model_out)):
            results_in = []
            results_out = []
            for j in range(len(self.model_out[i].nodes)):
                results_in.append(self.model_out[i].nodes[j]['attr_dict']['value'][0])
                results_out.append(self.model_out[i].nodes[j]['attr_dict']['value'][-1])
            initial_activation_levels.append(results_in)
            final_activation_level.append(results_out)
        return initial_activation_levels, final_activation_level


    def print_results(self, flt):
        "Print the initial and final activation levels of the nodes in the final graph"
        print("Activation levels of nodes in the FCMs")
        initial_activation_levels, final_activation_level = self.extract_weights()

        ling_initial = copy.deepcopy(initial_activation_levels)
        ling_final = copy.deepcopy(final_activation_level)
        for i in range(len(initial_activation_levels)):
            for j in range(len(initial_activation_levels[i])):
                ling_initial[i][j] = flt.get_linguisitic_term(ling_initial[i][j])
                ling_final[i][j] = flt.get_linguisitic_term(ling_final[i][j])

        for i in range(len(initial_activation_levels)):
            print(f"FCM {i}")
            print(f"\t Initial: {initial_activation_levels[i]}")
            print(f"\t\t  {ling_initial[i]}")
            print(f"\t Final: {final_activation_level[i]}")
            print(f"\t\t  {ling_final[i]}")


    def plot_al_values(self):
        "Plot the activation levels of the main node"
        G_main = self.model_out[0]

        len_nodes = len(G_main.nodes)
        central_node = G_main.nodes[len_nodes-1]
        y_values = central_node['attr_dict']['value']
        x_values = range(len(y_values))
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(x_values), y_values)
        plt.xlabel("Iterations")
        plt.ylabel("Activation Level")
        plt.title("Activation Level of Industry 4.0")
        plt.grid(True)
        plt.show(block=False)


def plot_al_values_graphs(fcms : dict, companies, colors):
    "Plot the activation levels of the main node of multiple FCM"
    all_x = {}
    all_y = {}
    for lambda_ in fcms.keys():
        fcm_objs : list[FCM] = fcms[lambda_]

        for i in range(len(companies)):
            G = fcm_objs[i].model_out[0]
            len_nodes = len(G.nodes)
            cnt_node = G.nodes[len_nodes-1]
            y_values = cnt_node['attr_dict']['value']
            x_values = list(range(len(y_values)))

            if i not in all_x.keys():
                all_x[i] = []
                all_y[i] = []
            all_x[i].append(x_values)
            all_y[i].append(y_values)

    plt.figure(figsize=(10, 5))
    lambdas = list(fcms.keys())
    for i in range(len(companies)):
        values_x = all_x[i]
        plt.subplot(1, len(companies), i+1)
        plt.grid(True)
        for j in range(len(values_x)):
            plt.plot(values_x[j], all_y[i][j], color=colors[j], label=f"lambda {lambdas[j]}")
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Activation Level")
        plt.title(f"Company {companies[i]}")

    plt.show(block=False)


def plot_sigmoid(lambda_values):
    plt.figure(figsize=(8, 6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors = colors[:len(lambda_values)]

    x = np.linspace(-5, 5, 100)
    for i in range(len(lambda_values)):
        lambda_value = lambda_values[i]
        y = []
        for j in x:
            y.append(FCM.sigmoid(j, lambda_value))
        plt.plot(x, y, color=colors[i], label=f'lambda = {lambda_value}')
        plt.axvline(x=-lambda_value, linestyle='--', color=colors[i])
        plt.axvline(x=lambda_value, linestyle='--', color=colors[i])

    plt.title(f'Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('Sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    return colors



if __name__ == "__main__":
    
    flt = FLT_class.define_al_fuzzy()
    flt.plot_triangle()

    flt2 = FLT_class.define_wm_fuzzy()
    flt2.plot_triangle()

    n_fcm = 6   # number of sub-fcms
    iterations = 100  # number of iterations
    threshold = 0.001
    print_status = False    # print status of the sub-fcms

    #lambda determina quanto il modello è sensibile ai cambiamenti dei AL
    #lamda grande -> più sensibile ai cambiamenti, tende a 0 o 1
    #lamda piccolo -> meno sensibile ai cambiamenti, tende a 0.5
    #lambdas = [0.8, 0.81]
    lambdas = [0.79] #0.79, 0.8, 0.81]
    colors = plot_sigmoid(lambdas)

    companies = ["low", "medium", "high", "mix"]   # AL file type of sub-fcms

    res = {}
    for lambda_value in lambdas:
        models = []
        for c in companies:
            print(f"Algorithm: Papageorgiou, Lambda: {lambda_value}, Iterations: {iterations}, Company Type: {c}")
            fcm_obj = FCM(n_fcm, iterations, c, flt)
            fcm_obj.run_fcm(lambda_value, threshold)
            fcm_obj.print_results(flt)
            linguistic_al = flt.get_linguisitic_term(fcm_obj.final_activation_level)
            print(f"Final activation level: {fcm_obj.final_activation_level} ({linguistic_al})")
            if print_status:
                fcm_obj.print_weights_nodes()
            models.append(fcm_obj)
            print("\n")
        res[str(lambda_value)] = models
    plot_al_values_graphs(res, companies, colors)
    plt.show()
