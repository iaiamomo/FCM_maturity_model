from fcm import FCM_class
import numpy as np
import json
import matplotlib.pyplot as plt
import fuzzy


def extract_weights(graph_list):
    initial_activation_levels = []
    final_activation_level = []
    for i in range(len(graph_list)):
        results_in = []
        results_out = []
        for j in range(len(graph_list[i].nodes)):
            results_in.append(graph_list[i].nodes[j]['attr_dict']['value'][0])
            results_out.append(graph_list[i].nodes[j]['attr_dict']['value'][-1])
        initial_activation_levels.append(results_in)
        final_activation_level.append(results_out)
    return initial_activation_levels, final_activation_level


def print_weights_nodes(graph_list, desc_nodes):
    for i in range(len(desc_nodes)):
        # grafo i-esimo
        to_print = f"FCM {i}\n"    
        for n in range(len(graph_list[i].nodes)):
            # nodo n-esimo
            to_print += f"\t{desc_nodes[i][str(n+1)]}:\t"
            to_print += f"{graph_list[i].nodes[n]['attr_dict']['value']}\n"
            pass
        print(to_print)


def print_results(initial_activation_levels, results_total):
    print("Activation levels of nodes in the FCMs")
    for i in range(len(initial_activation_levels)):
        print(f"FCM {i}")
        print(f"\t Initial: {initial_activation_levels[i]}")
        print(f"\t Final: {results_total[i]}")


def create_model(n_fcm, model_type=1, comp_type=1, iterations=25):
    graph_list = []
    desc_nodes_list = []
    desc_graphs = []
    for i in range(n_fcm):
        # get weights and activation levels from csv files 
        ww = np.genfromtxt(f'model{model_type}/{i}_wm.csv', delimiter=',')
        al = np.genfromtxt(f'model{model_type}/{i}_al_{comp_type}.csv', delimiter=',') if i != 0 else np.genfromtxt(f'model/{i}_al.csv', delimiter=',')

        # create graph from weights and activation levels
        h = FCM_class.fcm_from_matrix_to_graph(ww, al, 0, iterations+1)

        # get description of the graph
        desc = json.load(open(f'model{model_type}/{i}_desc.json'))
        desc_main = desc['main']
        desc_nodes = desc['nodes']
        desc_nodes_list.append(desc_nodes)
        desc_graphs.append(desc)
        #FCM_class.visualize(h, desc_main, desc_nodes)

        graph_list.append(h)

    return graph_list, desc_nodes_list, desc_graphs


def run_fcm(model, iterations, lambda_value, fcm_type="aa", flt=None):    
    # execute analysis of the FCM
    graph_list_out, final_activation_level = FCM_class.fcm_evaluate(model, iterations, lambda_value, fcm_type)
    assert graph_list_out is not None, "No algorithm selected"

    out_initial_al, out_final_al = extract_weights(graph_list_out)
    print_results(out_initial_al, out_final_al)

    ling_term = flt.get_linguisitic_term(final_activation_level)
    print("Final Activation Level:", final_activation_level, "(", ling_term, ")")

    return graph_list_out, final_activation_level


def plot_al_value(graph_list):
    G = graph_list[0]

    cnt_node = G.nodes[0]
    al_values = cnt_node['attr_dict']['value']
    iterations = len(al_values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), al_values)
    plt.xlabel("Iterations")
    plt.ylabel("Activation Level")
    plt.title("Activation Level of Industry 4.0")
    plt.grid(True)
    plt.show(block=False)


def plot_al_values_graphs(graphs, companies, colors):
    if type(graphs) is not dict:
        for i in range(len(companies)):
            G = graphs[i][0]
            len_nodes = len(G.nodes)
            cnt_node = G.nodes[len_nodes-1]
            al_values = cnt_node['attr_dict']['value']
            iterations = len(al_values)

            plt.subplot(1, len(companies), i+1)
            plt.plot(range(iterations), al_values)
            plt.ylim(0, 1)
            plt.xlabel("Iterations")
            plt.ylabel("Activation Level")
            plt.title(f"company {companies[i]}")
        
        plt.grid(True)
        plt.show(block=False)
    else:
        all_x = {}
        all_y = {}
        for lambda_ in graphs.keys():
            graph = graphs[lambda_]

            for i in range(len(companies)):
                G = graph[i][0]
                len_nodes = len(G.nodes)
                cnt_node = G.nodes[len_nodes-1]
                al_values = cnt_node['attr_dict']['value']
                iterations = len(al_values)

                if i not in all_x.keys():
                    all_x[i] = []
                    all_y[i] = []
                all_x[i].append(iterations)
                all_y[i].append(al_values)
        
        plt.figure(figsize=(10, 5))
        lambdas = list(graphs.keys())
        for i in range(len(companies)):
            values_x = all_x[i]
            plt.subplot(1, len(companies), i+1)
            plt.grid(True)
            for j in range(len(values_x)):
                plt.plot(range(values_x[j]), all_y[i][j], color=colors[j], label=f"lambda {lambdas[j]}")
            plt.ylim(0, 1)
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Activation Level")
            plt.title(f"company {companies[i]}")
        
        plt.show(block=False)


def plot_sigmoid(lambda_values):
    plt.figure(figsize=(8, 6))

    '''colors = []
    for i in range(len(lambda_values)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))'''

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors = colors[:len(lambda_values)]

    x = np.linspace(-5, 5, 100)
    for i in range(len(lambda_values)):
        lambda_value = lambda_values[i]
        y = []
        for j in x:
            y.append(FCM_class.sigmoid(j, lambda_value))
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
    flt = fuzzy.define_al_fuzzy()
    flt.plot_triangle()

    n_fcm = 6   # number of sub-fcms
    iterations = 25  # number of iterations
    #model -> modello originale e archi 0, 0.1, 0.5, 1
    #model2 -> pesi delle tecnologie sul sistema = 1/n_tecnologie e archi 0, 0.2, 0.5, 0.8
    #model3 -> pesi delle tecnologie sul sistema = 1 e archi 0, 0.2, 0.5, 0.8
    #model4 -> modello rivisto, pesi delle tecnologie sul sistem = 1/n_tecnologie e archi -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8
    model_type = 4  # model type
    fcm_algorithm = "papageorgiou"    # christoforou, papageorgiou, kosko, stylios
    print_status = False    # print status of the sub-fcms

    #lambda determina quanto il modello è sensibile ai cambiamenti dei AL
    #lamda grande -> più sensibile ai cambiamenti, tende a 0 o 1
    #lamda piccolo -> meno sensibile ai cambiamenti, tende a 0.5
    lambdas = [0.1, 0.8, 2, 2.5, 3, 3.5, 5]
    colors = plot_sigmoid(lambdas)

    #company_type 1 -> AL 0.1
    #company_type 2 -> AL misti
    #company_type 3 -> AL 0.9
    #company_type 4 -> AL 0
    companies = [1, 2, 3]   # AL file type of sub-fcms

    res = {}
    for lambda_value in lambdas:
        graphs = []
        for c in companies:
            print(f"Algorithm: {fcm_algorithm}, Lambda: {lambda_value}, Iterations: {iterations}, Company Type: {c}")
            model, desc_nodes_list, desc_graphs = create_model(n_fcm, model_type=model_type, comp_type=c, iterations=iterations)
            graph_list_out, final_activation_level = run_fcm(model, iterations, lambda_value, fcm_type=fcm_algorithm, flt=flt)
            if print_status:
                print_weights_nodes(graph_list_out, desc_nodes_list)
            graphs.append(graph_list_out)
            print("\n")
        res[str(lambda_value)] = graphs
    plot_al_values_graphs(res, companies, colors)
    plt.show()
