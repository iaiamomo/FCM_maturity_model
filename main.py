from fcm import FCM_class
import numpy as np
import json

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


def print_results(initial_activation_levels, results_total):
    print("Lamda value:", lambda_value)
    print("Activation levels of nodes in the FCMs")
    for i in range(len(initial_activation_levels)):
        print(f"FCM {i}")
        print(f"\t Initial: {initial_activation_levels[i]}")
        print(f"\t Final: {results_total[i]}")


def static_execute(graph_list, iterations, lambda_value, fcm_type="aa"):
    match fcm_type:
        case "cyprus":
            print("Cyprus")
            graph_list = FCM_class.cyprus_fcm_alg_graph(graph_list, g_index=0, start_iter=1, end_iter=iterations+1, lambda_value=lambda_value)
        case "kosko":
            print("Kosko")
            graph_list = FCM_class.kosko_alg_graph(graph_list, g_index=0, start_iter=1, end_iter=iterations+1, lambda_value=lambda_value)
        case "stylios":
            print("Stylios and Groumpos")
            graph_list = FCM_class.stylios_groumpos_alg_graph(graph_list, g_index=0, start_iter=1, end_iter=iterations+1, lambda_value=lambda_value)
        case _:
            graph_list = None
    return graph_list


def run_fcm(n_fcm, iterations, lambda_value, comp_type=1, fcm_type="aa"):
    # get all the files inside the subfolder and my_modeling folder using path
    graph_list = []
    for i in range(n_fcm):
        # get weights and activation levels from csv files 
        ww = np.genfromtxt(f'model/{i}_wm.csv', delimiter=',')
        al = np.genfromtxt(f'model/{i}_al_{comp_type}.csv', delimiter=',') if i != 0 else np.genfromtxt(f'model/{i}_al.csv', delimiter=',')

        # create graph from weights and activation levels
        h = FCM_class.fcm_from_matrix_to_graph(ww, al, 0, iterations+1)

        # get description of the graph
        desc = json.load(open(f'model/{i}_desc.json'))
        desc_main = desc['main']
        desc_nodes = desc['nodes']
        #FCM_class.visualize(h, desc_main, desc_nodes)

        graph_list.append(h)
    
    # execute static analysis on the sub-fcms
    graph_list_out = static_execute(graph_list, iterations, lambda_value, fcm_type)
    assert graph_list_out is not None, "No algorithm selected"

    out_initial_al, out_final_al = extract_weights(graph_list_out)
    print_results(out_initial_al, out_final_al)

    # extract the final activation level from the final graph
    n_main_concept = len(graph_list[0].nodes)-1
    final_activation_level = graph_list[0].nodes[n_main_concept]['attr_dict']['value'][-1]
    print("Final Activation Level:", final_activation_level)


if __name__ == "__main__":
    n_fcm = 6   # number of sub-fcms
    lambda_value = 0.2  # lambda value
    iterations = 1  # number of iterations
    company_type = 1    # al file type of sub-fcms
    # cyprus, kosko, stylios
    fcm_algorithm = "cyprus"    # fcm algorithm to use
    run_fcm(n_fcm, iterations, lambda_value, comp_type=company_type, fcm_type=fcm_algorithm)
