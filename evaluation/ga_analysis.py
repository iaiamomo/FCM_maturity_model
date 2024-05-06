import pickle
from utils.GA_class import *
from utils.FLT_class import *

def find_differences(arr1, arr2):
    indices = []

    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            indices.append(i)

    num_diff_elements = len(indices)

    return indices, num_diff_elements

cases_path = "cases"
company_types = ['low', 'mix']
results_tot = pickle.load(open('ga_results/results.pkl', 'rb'))
n_runs = 10
flt = define_al_fuzzy()

for company_type in company_types[1:]:
    n_fcm = 5

    initial_values = []
    for idx in range(1, n_fcm+1):
        al = pd.read_csv(f'../{cases_path}/{company_type}/{idx}_al.csv', header=None).values
        for x in range(1, len(al)):
            initial_values.append(flt.get_value(al[x][0]))

    differences = {}
    results, results_pop, results_gen, time_taken = results_tot[company_type]
    for i in range(n_runs):
        indices, num_differences = find_differences(initial_values, results[i])
        differences[i] = (indices, num_differences)
    
    for i in range(n_runs):
        print(f"run {i+1}: {differences[i][1]} differences")