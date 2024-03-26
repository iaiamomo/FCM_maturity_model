import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from random import randint
import copy
from utils.FLT_class import *
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
        for i in range(self.n_fcm):
            # get weights and activation levels from csv files 
            ww = np.genfromtxt(f'../model/{i}_wm.csv', delimiter=',')
            al = pd.read_csv(f'../cases/{company_type}/{i}_al.csv', header=None).values

            if len(new_values) > 0:
                # modify activation levels based on genes
                g_index = 0
                for x in range(len(al)):    # x is the row index
                    al[x][0] = new_values[g_index]
                    g_index+=1

            # create graph from weights and activation levels
            h = FCM.fcm_from_matrix_to_graph(ww, al, 0, self.iterations+1, self.al_flt)

            # get description of the graph
            desc = json.load(open(f'../model/{i}_desc.json'))
            desc_main = desc['main']
            desc_nodes = desc['nodes']
            desc_nodes_list.append(desc_nodes)
            self.desc_graphs.append(desc)
            #FCM_class.visualize(h, desc_main, desc_nodes)

            graph_list.append(h)

        return graph_list
    

    @staticmethod
    def fcm_from_matrix_to_graph(ww, al, depth, iterations, flt : Fuzzy_Linguistic_Terms):
        "Create a graph based on a given matrix"
        G = nx.DiGraph(depth=depth)
        n = ww.shape[0]

        # nodes
        for k in range(n):
            # "value" is an array representing the activation level through the iterations
            # "link" is the index of the graph to which the node is linked to
            G.add_node(k, attr_dict = {"value":[0]*iterations, "link":al[k][1]})
            l_al = al[k][0]
            v_al = flt.get_value(l_al)
            G.nodes[k]['attr_dict']['value'][0] = round(v_al, 5)

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


    def run_fcm(self, lambda_value, fcm_type, threshold=0.001):
        "Run the FCM algorithm"
        match fcm_type:
            case "papageorgiou":
                graph_list, t = FCM.papageorgiou_alg_graph(self.model[:], g_index=0, start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
            case "kosko":
                graph_list, t = FCM.kosko_alg_graph(self.model[:], g_index=0, start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value)
            case "stylios":
                graph_list, t = FCM.stylios_groumpos_alg_graph(self.model[:], g_index=0, start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value)
            case _:
                graph_list = None

        self.model_out = graph_list

        # refine the final activation level array
        for i in range (len(graph_list)):
            G = graph_list[i]
            for n in range(len(G.nodes)):
                G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
            graph_list[i] = G

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
                            print(f"Threshold reached at iteration {t}")
                            print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                            graph_list[g_index] = G
                            return graph_list, t
        
        graph_list[g_index] = G
        return graph_list, t
    

    @staticmethod
    def kosko_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value, threshold=0.001):
        "B. Kosko, 'Fuzzy cognitive maps', International Journal of Man-Machine Studies 24, p.p. 65-75, 1986."
        G = graph_list[g_index]

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph

                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list, t = FCM.kosko_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

                # contribution of the incoming edges of the node i (B_i^t)
                b = 0
                for edge in G.in_edges(node):   #for each incoming edge
                    # i is the node index
                    # j is the index of the node from which the edge comes
                    other, _ = edge
                    w_edge = G[other][node]['weight']   # edge weight
                    other_attr = G.nodes[other]['attr_dict']['value']   # other node attributes
                    
                    b += w_edge * other_attr[t-1]   # B_i^t = sum(w_ji * A_j^(t-1))
            
                final_al = round(FCM.sigmoid(b, lambda_value), 5)  # A_i^t = sigmoid(B_i^t)
                
                G.nodes[node]['attr_dict']['value'][t] = final_al

                if g_index == 0:
                    n_nodes = len(G.nodes)
                    if node == n_nodes-1:
                        if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                            print(f"Threshold reached at iteration {t}")
                            print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                            graph_list[g_index] = G
                            return graph_list, t
                
        graph_list[g_index] = G      
        return graph_list, t
    

    @staticmethod
    def stylios_groumpos_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value, threshold=0.001):
        "C.D. Stylios, P.P. Groumpos, 'A Soft Computing Approach for Modelling the Supervisor of Manufacturing Systems', Intelligent and Robotic Systems, vol. 26, p.p. 389-403, 1999."
        G = graph_list[g_index]
            
        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph

                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list, t = FCM.stylios_groumpos_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

                # contribution of the incoming edges
                b = 0
                for edge in G.in_edges(node):   #for each incoming edge
                    other, _ = edge
                    w_edge = G[other][node]['weight']
                    other_attr = G.nodes[other]['attr_dict']['value']
                    
                    b += w_edge * other_attr[t-1]   # B_i^t = sum(w_ij * A_j^(t-1))

                al_node = G.nodes[node]['attr_dict']['value'][t-1]

                b = b + al_node
            
                final_al = round(FCM.sigmoid(b, lambda_value), 5)

                G.nodes[node]['attr_dict']['value'][t] = final_al

                if g_index == 0:
                    n_nodes = len(G.nodes)
                    if node == n_nodes-1:
                        if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                            print(f"Threshold reached at iteration {t}")
                            print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
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


def plot_al_values_graphs(fcms : dict, companies, colors, algo):
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

    lambdas = list(fcms.keys())

    if algo == "papageorgiou":
        for i in range(len(companies)):
            plt.figure(algo+"_"+companies[i])
            xtick = 0
            values_x = all_x[i]
            plt.grid(True)
            for j in range(len(values_x)):
                plt.plot(values_x[j], all_y[i][j], color=colors[j], label=f"λ = {lambdas[j]}")
                if values_x[j][-1] > xtick:
                    xtick = values_x[j][-1]
            plt.ylim(0, 1)
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Activation Level")
            plt.title(f"Company {companies[i]}")
            plt.show(block=False)
    else:
        plt.figure(algo)
        x_tick = 0
        for i in range(len(lambdas)):
            low_x = all_x[0][i]
            low_y = all_y[0][i]
            high_x = all_x[1][i]
            high_y = all_y[1][i]
            plt.plot(low_x, low_y, color=colors[i], label=f"λ = {lambdas[i]}")
            plt.plot(high_x, high_y, color=colors[i], label=f"λ = {lambdas[i]}")
            x_val = low_x
            if len(low_x) > len(high_x):
                low_y = low_y[:len(high_x)]
                x_val = low_x[:len(high_x)]
            else:
                high_y = high_y[:len(low_x)]
                x_val = high_x[:len(low_x)]
            plt.fill_between(x_val, high_y, low_y, color=colors[i], alpha=0.2)
            if x_val[-1] > x_tick:
                x_tick = x_val[-1]
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        plt.xticks(range(0, x_tick+1))
        plt.xlabel("Iterations")
        plt.ylabel("Activation Level")
        plt.title("Company Low-High")
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
        plt.plot(x, y, color=colors[i], label=f'λ = {lambda_value}')
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
    flt = define_al_fuzzy()
    flt.plot_triangle()

    flt2 = define_wm_fuzzy()
    flt2.plot_triangle()

    n_fcm = 6
    iterations = 100
    threshold = 0.001
    fcm_algorithm = ["kosko", "stylios", "papageorgiou"]
    lambdas = [0.2, 0.5, 0.79, 1]
    colors = plot_sigmoid(lambdas)
    companies = ["low", "high"]

    for algo in fcm_algorithm:
        res = {}
        for lambda_value in lambdas:
            models = []
            for c in companies:
                print(f"Algorithm: {algo}, Lambda: {lambda_value}, Iterations: {iterations}, Company Type: {c}")
                fcm_obj = FCM(n_fcm, iterations, c, flt)
                fcm_obj.run_fcm(lambda_value, algo, threshold)
                fcm_obj.print_results(flt)
                linguistic_al = flt.get_linguisitic_term(fcm_obj.final_activation_level)
                print(f"Final activation level: {fcm_obj.final_activation_level} ({linguistic_al})")
                models.append(fcm_obj)
                print("\n")
            res[str(lambda_value)] = models
        plot_al_values_graphs(res, companies, colors, algo)
    plt.show()