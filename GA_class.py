import numpy as np
import matplotlib.pyplot as plt
import random
from FCM_class import FCM
import glob
import FLT_class
import pandas as pd
import json

# class representing an individual in the population
# in this case an individual is a FCM (activation levels of the nodes)
class Individual(object):

    def __init__(self, genes=None, target_val=0.6, individual_id = None, company_type='low'):
        if genes is None:
            al_files = glob.glob(f'cases/{company_type}/*_al.csv')
            n_fcm = len(al_files)
            flt = FLT_class.define_al_fuzzy()
            self.genes = []
            for i in range(1, n_fcm):
                al = pd.read_csv(f'cases/{company_type}/{i}_al.csv', header=None).values
                for x in range(1, len(al)):
                    v_al = flt.get_value(al[x][0])
                    self.genes.append(v_al)
            self.target_val = target_val
            self.fitness_val = 0
            self.id = individual_id
            self.algoritithm = None
        else:
            self.genes = genes
            self.target_val = target_val
            self.fitness_val = 0
            self.algoritithm = None


    # Returns fitness of individual
    # Fitness is the difference between target value and the calculated value
    def fitness(self, run=False, generation=None):
        if run==True:
            if generation == 0:
                self.algoritithm = Algorithm()  # run the FCM algorithm
            else:
                self.algoritithm = Algorithm(genes=self.genes)  # run the FCM algorithm
            computed_objective = self.algoritithm.result
            self.fitness_val = round(abs(self.target_val - computed_objective), 3)
        return self.fitness_val


# class representing the population
# in this case the population is an evolving set of individuals (FCMs with different activation levels)
class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=5, target_val=0.6, individual_size = 30, company_type='low'):
        self.pop_size = pop_size
        self.individuals_size = individual_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.target_val = target_val
        self.fitness_history = []
        self.parents = []
        self.elite = []
        self.done = False

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
        # set done to True if the target value is reached or if the fitness is too low
        if pop_fitness < 0.03 or self.individuals[0].fitness_val==0:
            self.done = True

        if generation is not None:
            print(f"Generation: {generation}, Population fitness: {pop_fitness}, Fitness sum: {fitness_sum}")


    # select the fittest individuals (elitist selection) to be the parents of next generation (lower fitness it better in this case)
    # also select non-fittest individuals (roulette wheel) to help get us out of local maximums
    def Select(self):
        # ELITIST SELECTION - keep the fittest as parents for next gen
        retain_length = (self.retain/100) * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]
        self.elite = self.parents[:] # keep a *copy* of the fittest individuals

        # ROULETTE WHEEL SELECTION - select some from unfittest and add to parents array
        # gives higher-probability chances to the better-performing individuals
        unfittest = self.individuals[int(retain_length):]
        max_fitness = sum(uf.fitness() for uf in unfittest) # sum of fitness of unfittest individuals
        while len(self.parents) < len(self.individuals):
            pick = random.uniform(0, max_fitness)
            current = 0
            for unfit in unfittest:
                current += unfit.fitness()  # here unfit.fitness() does not run the FCM algorithm
                if current < pick:
                    self.parents.append(unfit)
                    break


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
                child_genes = [random.choice(pixel_pair) for pixel_pair in zip(father.genes, mother.genes)] # randomly combines genes of parents
                child = Individual(genes=child_genes, target_val=self.target_val)
                children.append(child)
            self.individuals = children


    # alters at most one gene value for each individual
    def Mutation(self):
        for x in self.individuals:
            if self.mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(x.genes))
                new_value = np.random.randint(1,10)/10
                old_value = x.genes[mutate_index]
                while new_value < old_value:
                    new_value = np.random.randint(1,10)/10
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
    
    def __init__(self, genes=[]):
        flt = FLT_class.define_al_fuzzy()

        n_fcm = 6   # number of sub-fcms
        lambda_value = 0.79  # lambda value
        iterations = 100  # number of iterations
        threshold = 0.001    # threshold
        company_type = 'low'    # al file type of sub-fcms

        fcm_obj = FCM(n_fcm, iterations, company_type, flt, genes)
        fcm_obj.run_fcm(lambda_value, threshold)
        #fcm_obj.print_results(flt)
        self.result = fcm_obj.final_activation_level


class ALGE_class():

    # compute the number of genes in the FCM
    def compute_individual_size():
        individual_size = 0
        files = glob.glob("model/*_wm.csv")
        n_fcm = len(files)
        for i in range(1, n_fcm):
            filename = f"model/{i}_wm.csv"
            with open(filename) as f:
                line = f.readline()
                individual_size += len(line.split(","))-1
        return individual_size


    # execute the what-if analysis of the FCM
    def what_if(n_runs, pop_size, generation, mutate_prob, retain, target_val, company_type):
        # FCM parameters    
        individual_size = ALGE_class.compute_individual_size()

        results = []
        results_pop = []
        results_gen = []
        for k in range(n_runs):
            pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, target_val=target_val, individual_size=individual_size, company_type=company_type)
        
            for x in range(generation):
                pop.grade(generation = x)
                if pop.done or x == generation - 1: # if target value is reached or if we reached the last generation
                    print(f"\tSimulation: {k} Finished at generation: {x}, Population fitness: {pop.fitness_history[-1]}")
                    break
                pop.evolve()

            finalAL = pop.individuals[0].genes[:]   # copy of the activation levels of the best individual
            results.append(finalAL) # add the activation levels of the best individual to the results array
            results_pop.append(pop) # add the population to the results array
            results_gen.append(x)   # add the generation to the results array

        return results, results_pop, results_gen


    def visualize(results, results_pop, results_gen, target_val):
        for i in range(len(results)):
            al_ = results[i]
            pop_ = results_pop[i]
            gen_ = results_gen[i]
            
            # Plot fitness history
            print("Showing fitness history graph")
            plt.plot(np.arange(len(pop_.fitness_history)), pop_.fitness_history, label=f'Sim: {i}')

        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title(f'Target Value: {target_val}')
        plt.legend()


if __name__ == "__main__":
    # genetic algorithm parameters
    n_runs = 1  # number of simulations
    pop_size = 50   # number of individuals (FCM) in the population
    generation = 250    # number of generations
    mutate_prob = 0.5   # probability of mutation
    retain = 5  # percentage of fittest individuals to be kept as parents for next generation (elitist selection)
    flt = FLT_class.define_al_fuzzy()

    # target concept value
    target_val = 0.8
    company_type = 'low'

    results, results_pop, results_gen = ALGE_class.what_if(n_runs, pop_size, generation, mutate_prob, retain, target_val, company_type)

    # from the results_pop array, get the best individual
    best_individuals = []
    for pop in results_pop:
        best_individuals.append(pop.individuals[0])

    # print the best individual
    best_best_individual = max(best_individuals, key=lambda x: x.algoritithm.result)
    print(f"Best individual: {best_best_individual.genes}")
    model_files = glob.glob(f'model/*_desc.json')
    n_fcm = len(model_files)
    idx_best_individual = 0
    for i in range(1, n_fcm):
        fcm_desc = f"{i}_desc.json"
        with open(f'model/{fcm_desc}') as json_file:
            data = json.load(json_file)
            print(f"FCM {data['main']}")
            n_nodes = len(data['nodes'])
            with open(f'cases/{company_type}/{i}_al.csv') as al_file:
                al = pd.read_csv(al_file, header=None).values
                for j in range(1, n_nodes):
                    initial_al = al[j][0]
                    final_al = flt.get_linguisitic_term(best_best_individual.genes[idx_best_individual])
                    idx_best_individual += 1
                    print(f"\tNode {data['nodes'][str(j+1)]}")
                    print(f"\t\tInitial AL:\t{initial_al}")
                    print(f"\t\tFinal AL: \t{final_al}")

    # visualize the results
    ALGE_class.visualize(results, results_pop, results_gen, target_val)

    plt.show()