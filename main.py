from fcm import FCM_class
import numpy as np

def extract_weights(graph_list):
    initial_activation_levels = []
    for i in range(len(graph_list)):
        results = []
        for j in range(len(graph_list[i].nodes)):
            results.append(graph_list[i].nodes[j]['attr_dict']['value'][0])
        initial_activation_levels.append(results)
        
    results_total = []
    for i in range(len(graph_list)):
        results = []
        for j in range(len(graph_list[i].nodes)):
            results.append(graph_list[i].nodes[j]['attr_dict']['value'][iterations])
        results_total.append(results)

    return initial_activation_levels, results_total


def print_results(initial_activation_levels, results_total):
    print("Lamda value:", lambda_value)
    for i in range(len(initial_activation_levels)):
        print("al_in FCM", i, initial_activation_levels[i])
        print("al_out FCM", i, results_total[i])
    print()


def static_execute(graph_list, iterations, lambda_value, fcm_type="aa"):
    match fcm_type:
        case "aa":
            print("ANDREAS")
            graph_list = FCM_class.aa_fcm_alg_graph(graph_list, 1, iterations+1, lambda_value)
        case "kosko":
            print("KOSKO")
            graph_list = FCM_class.kosko_alg_graph(graph_list, 1, iterations+1, lambda_value)
        case "stylios":
            print("STYLIOS")
            graph_list = FCM_class.stylios_groumpos_alg_graph(graph_list, 1, iterations+1, lambda_value)
        case _:
            print("ANDREAS")
            graph_list = FCM_class.aa_fcm_alg_graph(graph_list, 1, iterations+1, lambda_value)
    return graph_list


def run_fcm(n_fcm, iterations, lambda_value, comp_type=1, fcm_type="aa"):
    # get all the files inside the subfolder and my_modeling folder using path
    graph_list = []
    for i in range(1, n_fcm):
        # get weights and activation levels from csv files 
        ww = np.genfromtxt(f'model/{i}_wm.csv', delimiter=',')
        al = np.genfromtxt(f'model/{i}_al_{comp_type}.csv', delimiter=',')

        h = FCM_class.fcm_from_matrix_to_graph(ww, al, 0, iterations+1)
        graph_list.append(h)
    
    graph_list_1 = static_execute(graph_list, iterations, lambda_value, fcm_type) 
    initial_activation_levels, results_total = extract_weights(graph_list_1)
    print_results(initial_activation_levels, results_total)

    # main fcm
    f_csv = open("res_al.csv", "w")
    for i in range(len(results_total)):
        value = results_total[i][-1]
        f_csv.write(f"{value}\n")
    f_csv.write("0\n")
    f_csv.close()

    ww = []
    for i in range(len(results_total)):
        r_ww = [0] * len(results_total)
        r_ww.append(1)
        ww.append(r_ww)
    ww.append([0] * (len(results_total) + 1))
    ww = np.array(ww)
    np.savetxt(f'res_ww.csv', ww, delimiter=',', fmt='%d')
    al = np.genfromtxt(f'res_al.csv', delimiter=',')

    h = FCM_class.fcm_from_matrix_to_graph(ww, al, 0, iterations+1)
    final_graph = [h]

    final_graph_res = static_execute(final_graph, iterations, lambda_value, fcm_type)
    final_initial_activation_levels, final_results_total = extract_weights(final_graph_res)
    print_results(final_initial_activation_levels, final_results_total)

    final_activation_level = final_results_total[0][-1]
    print("Final Activation Level:", final_activation_level)


if __name__ == "__main__":
    n_fcm = 5
    lambda_value = 0.8
    iterations = 5
    run_fcm(n_fcm, iterations, lambda_value, comp_type=1, fcm_type="aa")
