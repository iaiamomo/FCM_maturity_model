from utils.FCM_class import FCM
from utils.FLT_class import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import time
import random
import glob

model_path = "../model"
cases_path = "../cases"

# class representing an individual in the population
# in this case an individual is a FCM (activation levels of the nodes)
class Individual(object):

    def __init__(self, genes=None, target_val=0.6, individual_id = None, company_type='low'):
        if genes is None:
            al_files = glob.glob(f'{cases_path}/{company_type}/*_al.csv')
            n_fcm = len(al_files)
            flt = define_al_fuzzy()
            self.genes = []
            for i in range(1, n_fcm):
                al = pd.read_csv(f'{cases_path}/{company_type}/{i}_al.csv', header=None).values
                for x in range(1, len(al)):
                    v_al = flt.get_value(al[x][0])
                    self.genes.append(v_al)
            self.id = individual_id
        else:
            self.genes = genes
        self.target_val = target_val
        self.fitness_val = 0
        self.algoritithm = None
        self.company_type = company_type


    # Returns fitness of individual
    # Fitness is the difference between target value and the calculated value
    def fitness(self, run=False, generation=None):
        if run==True:
            if generation == 0:
                self.algoritithm = Algorithm(company_type=self.company_type)  # run the FCM algorithm
            else:
                self.algoritithm = Algorithm(genes=self.genes, company_type=self.company_type)  # run the FCM algorithm
            computed_objective = self.algoritithm.result
            self.fitness_val = round(abs(self.target_val - computed_objective), 3)
        return self.fitness_val


# class representing the population
# in this case the population is an evolving set of individuals (FCMs with different activation levels)
class Population(object):

    def __init__(self, pop_size=10, crossover_prob=0.7, retain=5, target_val=0.6, individual_size = 30, company_type='low', flt:Fuzzy_Linguistic_Terms=None):
        self.pop_size = pop_size
        self.individuals_size = individual_size
        self.crossover_prob = crossover_prob
        self.retain = retain
        self.target_val = target_val
        self.fitness_history = []
        self.ind_fitness_history = []
        self.parents = []
        self.elite = []
        self.done = False
        self.values = [flt.get_value(x) for x in flt.linguistic_terms.keys()][1:]

        # Create individuals
        self.individuals : list[Individual] = []
        for x in range(pop_size):
            self.individuals.append(Individual(genes=None, target_val=self.target_val, individual_id=x, company_type=company_type))


    # Grade the generation by getting the average fitness of its individuals
    def grade(self, generation=None):
        fitness_sum = 0
        for x in self.individuals:
            fitness_sum += x.fitness(run=True, generation=generation)  # run the FCM algorithm
        fitness_sum = round(fitness_sum, 3)

        pop_fitness = round(fitness_sum / self.pop_size, 3)
        self.fitness_history.append(pop_fitness)

        # sort individuals by fitness (lower fitness it better in this case)
        self.individuals = sorted(self.individuals, key=lambda x: x.fitness())  # here x.fitness() does not run the FCM algorithm
        self.ind_fitness_history.append(self.individuals[0].fitness_val)

        # set done to True if the target value is reached or if the fitness is too low
        if pop_fitness < 0.03 or self.individuals[0].fitness_val==0 or self.individuals[0].fitness_val < 0.03:
            self.done = True

        if generation is not None:
            print(f"Generation: {generation}, Population fitness: {pop_fitness}, Fitness sum: {fitness_sum}, Best individual fitness: {self.individuals[0].fitness_val}")


    # select the fittest individuals (elitist selection) to be the parents of next generation (lower fitness it better in this case)
    def Select(self):
        # ELITIST SELECTION - keep the fittest as parents for next gen
        retain_length = (self.retain/100) * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]
        self.elite = self.parents[:] # keep a *copy* of the fittest individuals


    # crossover the parents to generate a new generation of individuals
    def Crossover(self):
        target_children_size = self.pop_size
        children = self.elite[:] # copy of the fittest are kept as children
        if len(self.parents) > 0:
            while len(children) < target_children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                while mother == father:
                    mother = random.choice(self.parents)
                # one-point crossover
                if self.crossover_prob > np.random.rand():
                    idx_crossover = np.random.randint(0, len(father.genes))
                    child_genes = father.genes[:idx_crossover] + mother.genes[idx_crossover:]
                    child = Individual(genes=child_genes, target_val=self.target_val)
                    children.append(child)
                    child_genes = mother.genes[:idx_crossover] + father.genes[idx_crossover:]
                    child = Individual(genes=child_genes, target_val=self.target_val)
                    children.append(child)
            children = children[:target_children_size]
            self.individuals = children


    # alters at most one gene value for each individual
    def Mutation(self):
        for x in self.individuals:
            # generate a dictionary of position, gene
            dic_genes = {i: x.genes[i] for i in range(len(x.genes))}
            # get the position of the lower gene value
            mutate_index = min(dic_genes, key=dic_genes.get)
            to_mutate = x.genes[mutate_index]
            dic_genes_to_mutate = {i: x.genes[i] for i in range(len(x.genes)) if x.genes[i] == to_mutate}
            # get a random position to mutate
            mutate_index = random.choice(list(dic_genes_to_mutate.keys()))
            # mutate the gene to a random value greater than the current value
            new_value = np.random.randint(1,10)/10
            old_value = x.genes[mutate_index]
            while new_value <= old_value:
                new_value = np.random.choice(self.values)
                if new_value == 0.9:
                    break
            x.genes[mutate_index] = new_value


    # evolves the current population to the next generation
    def evolve(self):
        # 1. Selection
        self.Select()
        # 2. Crossover
        self.Crossover()
        # 3. Mutation
        self.Mutation()
        
        # Reset parents and children
        self.parents = []
        self.children = []
        self.elite = []


# class representing the FCM algorithm
class Algorithm(object):
    
    def __init__(self, genes=[], company_type='low'):
        flt = define_al_fuzzy()

        n_fcm = 5   # number of sub-fcms
        lambdas = {
            1: 0.83,
            2: 0.85,
            3: 0.81,
            4: 0.91,
            5: 0.735
        }
        iterations = 100  # number of iterations
        threshold = 0.001    # threshold

        fcm_obj = FCM(n_fcm, iterations, lambdas, company_type, flt, genes)
        fcm_obj.run_fcm(threshold)
        #fcm_obj.print_results(flt)
        self.result = fcm_obj.main_final_al


class ALGA_class():

    # compute the number of genes in the FCM
    def compute_individual_size():
        individual_size = 0
        files = glob.glob(f"{model_path}/*_wm.csv")
        n_fcm = len(files)
        for i in range(1, n_fcm):
            filename = f"{model_path}/{i}_wm.csv"
            with open(filename) as f:
                line = f.readline()
                individual_size += len(line.split(","))-1
        return individual_size


    # execute the what-if analysis of the FCM
    def what_if(n_runs, pop_size, generation, retain, target_val, company_type, flt):
        # FCM parameters    
        individual_size = ALGA_class.compute_individual_size()

        results = []
        results_pop = []
        results_gen = []
        for k in range(n_runs):
            pop = Population(pop_size=pop_size, retain=retain, target_val=target_val, individual_size=individual_size, company_type=company_type, flt=flt)
        
            for x in range(generation):
                pop.grade(generation = x)
                if pop.done or x == generation - 1: # if target value is reached or if we reached the last generation
                    print(f"\tSimulation: {k} Finished at generation: {x}, Population fitness: {pop.fitness_history[-1]}, Individual fitness: {pop.ind_fitness_history[-1]}")
                    break
                pop.evolve()

            finalAL = pop.individuals[0].genes[:]   # copy of the activation levels of the best individual
            results.append(finalAL) # add the activation levels of the best individual to the results array
            results_pop.append(pop) # add the population to the results array
            results_gen.append(x)   # add the generation to the results array

        return results, results_pop, results_gen


    def visualize(results, results_pop, company_type):
        plt.figure(f"fitness_{company_type}")
        max_generations = max([len(pop_.ind_fitness_history) for pop_ in results_pop])
        for i in range(len(results)):
            pop_ = results_pop[i]
            # Plot fitness history
            plt.plot(np.arange(len(pop_.ind_fitness_history)), pop_.ind_fitness_history, label=f'Run: {i}')
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.xticks(np.arange(0, max_generations, 2))
        plt.legend()
        plt.grid()
        plt.title("Fitness of the best individuals")
        plt.show(block=False)

        plt.figure(f"pop_fitness_{company_type}")
        max_generations = max([len(pop_.fitness_history) for pop_ in results_pop])
        for i in range(len(results)):
            pop_ = results_pop[i]
            # Plot fitness history
            plt.plot(np.arange(len(pop_.fitness_history)), pop_.fitness_history, label=f'Run: {i}')
        plt.ylabel('Pop Fitness')
        plt.xlabel('Generations')
        plt.xticks(np.arange(0, max_generations, 2))
        plt.legend()
        plt.grid()
        plt.title("Fitness of the population")
        plt.show(block=False)


if __name__ == "__main__":
    # genetic algorithm parameters
    n_runs = 10  # number of simulations
    pop_size = 50   # number of individuals (FCM) in the population
    generation = 250    # number of generations
    best = False    # print the best individual
    crossover_prob = 0.7    # probability of crossover
    retain = 15  # percentage of fittest individuals to be kept as parents for next generation (elitist selection)
    company_types = ['low', 'mix']
    target_val = 'VH'
    flt = define_al_fuzzy()
    target_val = flt.get_value(target_val)

    results_tot = {}
    for company_type in company_types:
        print(f"Running GA for {company_type} company type, crossover: {crossover_prob}, retain: {retain}")
        time_1 = time.time()
        results, results_pop, results_gen = ALGA_class.what_if(n_runs, pop_size, generation, retain, target_val, company_type, flt)
        time_2 = time.time()
        time_taken = time_2 - time_1
        print(f"Time taken: {time_taken} sec")

        # from the results_pop array, get the best individuals
        best_individuals = []
        for pop in results_pop:
            best_individuals.append(pop.individuals[0])

        # print the best individual
        model_files = glob.glob(f'{model_path}/*_desc.json')
        n_fcm = len(model_files)
        for i in range(len(best_individuals)):
            best_best_individual = best_individuals[i]
            idx_best_individual = 0
            file_string = ""
            for j in range(1, n_fcm):
                fcm_desc = f"{j}_desc.json"
                with open(f'{model_path}/{fcm_desc}') as json_file:
                    data = json.load(json_file)
                    print(f"FCM {data['main']}")
                    file_string += f"FCM {data['main']}\n"
                    n_nodes = len(data['nodes'])
                    with open(f'{cases_path}/{company_type}/{j}_al.csv') as al_file:
                        al = pd.read_csv(al_file, header=None).values
                        for k in range(1, n_nodes):
                            initial_al = al[k][0]
                            final_al = flt.get_linguisitic_term(best_best_individual.genes[idx_best_individual])
                            idx_best_individual += 1
                            print(f"\tNode {data['nodes'][str(k+1)]}")
                            print(f"\t\tInitial AL:\t{initial_al}")
                            print(f"\t\tFinal AL: \t{final_al}")
                            file_string += f"\tNode {data['nodes'][str(k+1)]}\n"
                            file_string += f"\t\tInitial AL:\t{initial_al}\n"
                            file_string += f"\t\tFinal AL: \t{final_al}\n"
            with open(f'ga_results/{company_type}_best_{i}_results.txt', 'w') as file:
                file.write(file_string)

        results_tot[company_type] = (results, results_pop, results_gen, time_taken)
        # Plot results
        ALGA_class.visualize(results, results_pop, company_type)
    
    # Save results in a pickle file
    with open('ga_results/results.pkl', 'wb') as f:
        pickle.dump(results_tot, f)

    plt.show()
