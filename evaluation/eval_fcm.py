import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from random import randint
import copy
import utils.FLT_class as FLT_class
import pandas as pd

model_path = "../model"
cases_path = "../cases"
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
other_colors = ['gray', 'orange', 'hotpink']
lambdas_lab = ['low', 'high']

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


    def run_fcm(self, fcm_type, threshold=0.001):
        self.model_out = []
        self.final_al = []

        "Run the FCM algorithm"
        match fcm_type:
            case "papageorgiou":
                for i in range(len(self.model)):
                    lambda_value = self.lambdas[i+1]
                    G, t = FCM.papageorgiou_alg_graph(self.model[i], start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
                    for n in range(len(G.nodes)):
                        G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
                    self.model_out.append(G)
            case "kosko":
                for i in range(len(self.model)):
                    lambda_value = self.lambdas[i+1]
                    G, t = FCM.kosko_alg_graph(self.model[i], start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
                    for n in range(len(G.nodes)):
                        G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
                    self.model_out.append(G)
            case "stylios":
                for i in range(len(self.model)):
                    lambda_value = self.lambdas[i+1]
                    G, t = FCM.stylios_groumpos_alg_graph(self.model[i], start_iter=1, end_iter=self.iterations+1, lambda_value=lambda_value, threshold=threshold)
                    for n in range(len(G.nodes)):
                        G.nodes[n]['attr_dict']['value'] = G.nodes[n]['attr_dict']['value'][:t]
                    self.model_out.append(G)
            case _:
                G = None

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

        print(max_iter)
        self.final_al_mean = []
        for i in range(max_iter):
            mean_j = []
            for j in range(len(self.model_out)):
                mean_j.append(self.model_out[j].nodes[0]['attr_dict']['value'][i])
            self.final_al_mean.append(np.mean(mean_j))
        self.main_final_al = self.final_al_mean[-1]
        print(f"Main final activation level: {self.main_final_al}")

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
                        print(f"Threshold reached at iteration {t}")
                        print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                        return G, t

        return G, t


    @staticmethod
    def kosko_alg_graph(graph, start_iter, end_iter, lambda_value, threshold=0.001):
        "B. Kosko, 'Fuzzy cognitive maps', International Journal of Man-Machine Studies 24, p.p. 65-75, 1986."
        G = graph

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph
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

                if node == 0:
                    if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                        print(f"Threshold reached at iteration {t}")
                        print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                        return G, t

        return G, t


    @staticmethod
    def stylios_groumpos_alg_graph(graph, start_iter, end_iter, lambda_value, threshold=0.001):
        "C.D. Stylios, P.P. Groumpos, 'A Soft Computing Approach for Modelling the Supervisor of Manufacturing Systems', Intelligent and Robotic Systems, vol. 26, p.p. 389-403, 1999."
        G = graph

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph
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

                if node == 0:
                    if abs(G.nodes[node]['attr_dict']['value'][t] - G.nodes[node]['attr_dict']['value'][t-1]) < threshold:
                        print(f"Threshold reached at iteration {t}")
                        print(f"Node: {node}, Value: {G.nodes[node]['attr_dict']['value'][t]}, prev: {G.nodes[node]['attr_dict']['value'][t-1]}")
                        return G, t

        return G, t


    @staticmethod
    def sigmoid(x, lambda_value):
        "Sigmoid function"
        return 1/(1+np.exp(-lambda_value*x))


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


def plot_al_values_graphs(fcms : dict, companies, colors, algorithms, lambdas_values):
    "Plot the activation levels of the main node of multiple FCM"

    for algo in algorithms[:2]:
        fcms = res_algo[algo]
        all_x = {}
        all_y = {}
        for i in range(len(lambdas_values)):
            if i == 0:
                continue
            fcm_objs = fcms[i]
            for j in range(len(companies)):
                models = fcm_objs[j]
                if companies[j] not in all_y:
                    all_y[companies[j]] = []
                all_y[companies[j]].append(models.final_al_mean)
        name_algorithm = algo[0].upper() + algo[1:]
        plt.figure(name_algorithm)
        plt.title(name_algorithm)
        for i in range(len(companies)):
            comp_type = companies[i]
            all_x = min(len(all_y[comp_type][0]), len(all_y[comp_type][1]))
            plt.plot(list(range(len(all_y[comp_type][0]))), all_y[comp_type][0], color=colors[i], label=f"Company {comp_type}")
            plt.plot(list(range(len(all_y[comp_type][1]))), all_y[comp_type][1], color=colors[i])
            plt.fill_between(list(range(all_x)), all_y[comp_type][0][:all_x], all_y[comp_type][1][:all_x], color=colors[i], alpha=0.2)
        plt.xlabel("Iterations")
        plt.ylabel("Activation Level")
        plt.legend()
        plt.grid()
        plt.xticks(range(all_x+1))
        plt.show(block=False)

    fcms = res_algo[algorithms[2]]
    all_x = {}
    all_y = {}
    fcm_objs = fcms[0]
    for j in range(len(companies)):
        models = fcm_objs[j]
        if companies[j] not in all_y:
            all_y[companies[j]] = []
        all_y[companies[j]].append(models.final_al_mean)
    for i in range(len(companies)):
        company = companies[i][0].upper() + companies[i][1:]
        plt.figure(company)
        plt.title(f"Company {company}")
        comp_type = companies[i]
        all_x = len(all_y[comp_type][0])
        models = fcm_objs[i]
        for j in range(len(models.model_out)):
            fcm_j = models.model_out[j]
            plt.plot(list(range(len(fcm_j.nodes[0]['attr_dict']['value']))),fcm_j.nodes[0]['attr_dict']['value'], color=colors[j+1], linestyle='-.', label=f"FCM IT{j+1}")
        plt.plot(list(range(len(all_y[comp_type][0]))), all_y[comp_type][0], color=colors[0], linewidth=2, label=f"M0 - λ mix")
        for j in range(1, len(lambdas_values)):
            fcm_objs_ = fcms[j]
            models_ = fcm_objs_[i]
            y_val = models_.final_al_mean
            plt.plot(list(range(len(y_val))), y_val, color=other_colors[j], label=f"M0 - λ {lambdas_lab[j-1]}")
        plt.ylim(0, 1.04)
        plt.grid()
        plt.xlabel("Iterations")
        plt.ylabel("Activation Level")
        plt.legend()
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
    flt = FLT_class.define_al_fuzzy()
    flt.plot_triangle()

    flt2 = FLT_class.define_wm_fuzzy()
    flt2.plot_triangle()

    n_fcm = 5
    iterations = 100
    threshold = 0.001
    fcm_algorithm = ["kosko", "stylios", "papageorgiou"]
    lambdas_values = [{
            1: 0.83,
            2: 0.85,
            3: 0.81,
            4: 0.91,
            5: 0.735
        }, {
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2,
            5: 0.2
        }, {
            1: 2,
            2: 2,
            3: 2,
            4: 2,
            5: 2
        }
    ]
    plot_sigmoid(lambdas_values[0])
    companies = ["low", "high"]

    res_algo = {}
    for algo in fcm_algorithm:
        res = []
        for i in range(len(lambdas_values)):
            lambdas = lambdas_values[i]
            models = []
            for c in companies:
                print(f"Algorithm: {algo}, Iterations: {iterations}, Company Type: {c}")
                fcm_obj = FCM(n_fcm, iterations, lambdas, c, flt)
                fcm_obj.run_fcm(algo, threshold)
                fcm_obj.print_results(flt)
                linguistic_al = flt.get_linguisitic_term(fcm_obj.main_final_al)
                print(f"Final activation level: {fcm_obj.main_final_al} ({linguistic_al})")
                models.append(fcm_obj)
                print("\n")
            res.append(models)
        res_algo[algo] = res
    plot_al_values_graphs(res, companies, colors, fcm_algorithm, lambdas_values)
    plt.show()