import numpy as np
import matplotlib.pyplot as plt
import random
from fcm import FCM_class
import glob

# class representing an individual in the population
# in this case an individual is a FCM (activation levels of the nodes)
class Individual(object):

    def __init__(self, genes=None, individual_size = 20, target_val=0.6, individual_id = None):
        if genes is None:
            self.genes = np.random.randint(1, 10, size=individual_size) / 10    # random values between 0.1 and 0.9
            self.target_val = target_val
            self.fitness_val = 0
            self.id = individual_id
            self.algoritithm = None
        else:
            self.fitness_val = 0
            self.genes = genes
            self.target_val = target_val
            self.algoritithm = None


    # Returns fitness of individual
    # Fitness is the difference between target value and the calculated value
    def fitness(self, run=False):
        if run==True:
            self.algoritithm = Algorithm(genes=self.genes)  # run the FCM algorithm
            computed_objective = self.algoritithm.result
            self.fitness_val = round(abs(self.target_val - computed_objective), 3)
        return self.fitness_val


# class representing the population
# in this case the population is an evolving set of individuals (FCMs with different activation levels)
class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=5, target_val=0.6, individual_size = 30):
        self.pop_size = pop_size
        self.individuals_size = individual_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.target_val = target_val
        self.fitness_history = []
        self.parents = []
        self.elit = []
        self.done = False

        # Create individuals
        self.individuals : list[Individual] = []
        for x in range(pop_size):
            self.individuals.append(Individual(genes=None, individual_size=self.individuals_size, target_val=self.target_val, individual_id=x))

    # Grade the generation by getting the average fitness of its individuals
    def grade(self, generation=None):
        fitness_sum = 0
        for x in self.individuals:
            fitness_sum += x.fitness(run=True)  # run the FCM algorithm
        
        pop_fitness = round(fitness_sum / self.pop_size, 3)
        self.fitness_history.append(pop_fitness)

        # sort individuals by fitness (lower fitness it better in this case)
        self.individuals = sorted(self.individuals, key=lambda x: x.fitness())  # here x.fitness() does not run the FCM algorithm
        # set done to True if the target value is reached or if the fitness is too low
        if pop_fitness < 0.02 or self.individuals[0].fitness_val==0:
            #pop_fitness = 0
            #self.fitness_history.append(pop_fitness)
            self.done = True
            
        if generation is not None:
            print(f"Generation: {generation}, Population fitness: {pop_fitness}, Fitness sum: {fitness_sum}")

    
    # select the fittest individuals (elitist selection) to be the parents of next generation (lower fitness it better in this case)
    # also select non-fittest individuals (roulette wheel) to help get us out of local maximums
    def Select(self):
        # ELITIST SELECTION - keep the fittest as parents for next gen
        retain_length = (self.retain/100) * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]
        self.elit = self.parents[:] # keep a *copy* of the fittest individuals

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
        children = self.elit[:] # copy of the fittest are kept as children
        if len(self.parents) > 0:
            while len(children) < target_children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                #if father != mother:
                child_genes = [random.choice(pixel_pair) for pixel_pair in zip(father.genes, mother.genes)] # randomly combines genes of parents
                child = Individual(genes=child_genes, individual_size=self.individuals_size, target_val=self.target_val)
                children.append(child)
            self.individuals = children


    # alters at most one gene value for each individual
    def Mutation(self):
        for x in self.individuals:
            if self.mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(x.genes))
                x.genes[mutate_index] = np.random.randint(1,10)/10


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
        self.elit = []


# class representing the FCM algorithm
class Algorithm(object):
    
    def __init__(self, genes=[]):
        graph_list = []
        n_fcm = 6   # number of sub-fcms
        lambda_value = 2  # lambda value
        iterations = 25  # number of iterations
        company_type = 1    # al file type of sub-fcms
        model_type = 4  # model type
        fcm_algorithm = "papageorgiou"    # christoforou, papageorgiou, kosko, stylios
    
        for i in range(n_fcm):
            # get weights and activation levels from csv files
            ww = np.genfromtxt(f'model{model_type}/{i}_wm.csv', delimiter=',')
            al = np.genfromtxt(f'model{model_type}/{i}_al_{company_type}.csv', delimiter=',') if i != 0 else np.genfromtxt(f'model/{i}_al.csv', delimiter=',')

            # modify activation levels based on genes
            g_index = 0
            for x in range(len(al)):    # x is the row index
                al[x][0] = genes[g_index]
                g_index+=1

            h = FCM_class.fcm_from_matrix_to_graph(ww, al, 0, iterations+1)
            graph_list.append(h)
    
        graph_list, final_activation_level = FCM_class.fcm_evaluate(graph_list, iterations, lambda_value, fcm_algorithm)
        self.result = final_activation_level
        #self.result.append(abs(graph_list[0].nodes[11]['attr_dict']['value'][iterations]-0.9))
        #self.result.append(abs(graph_list[0].nodes[6]['attr_dict']['value'][iterations]-0.7))


class ALGE_class():

    # compute the number of genes in the FCM
    def compute_individual_size():
        individual_size = 0
        files = glob.glob("model/*_wm.csv")
        for file in files:
            with open(file) as f:
                line = f.readline()
                individual_size += len(line.split(","))-1
        n_fcm = len(files)
        individual_size -= n_fcm - 1    # remove the common concepts between FCMs
        individual_size -= 1    # remove the target concept
        return individual_size
    

    # execute the what-if analysis of the FCM
    def what_if(n_runs, pop_size, generation, mutate_prob, retain, target_val):
        # FCM parameters    
        individual_size = ALGE_class.compute_individual_size()

        results = []
        results_pop = []
        results_gen = []
        for k in range(n_runs):
            pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, target_val=target_val, individual_size=individual_size)
        
            for x in range(generation):
                pop.grade(generation = x)
                if pop.done or x == generation - 1: # if target value is reached or if we reached the last generation
                    print(f"Simulation: {k} Finished at generation: {x}, Population fitness: {pop.fitness_history[-1]}")
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
            plt.plot(np.arange(len(pop_.fitness_history)), pop_.fitness_history)
            plt.ylabel('Fitness')
            plt.xlabel('Generations')
            plt.xlim(0, gen_)
            plt.ylim(0.0, 0.3)
            plt.title(f'Target Value: {target_val}')
            plt.show()


if __name__ == "__main__":
    # genetic algorithm parameters
    n_runs = 5  # number of simulations
    pop_size = 50   # number of individuals (FCM) in the population
    generation = 250    # number of generations
    mutate_prob = 0.1   # probability of mutation
    retain = 5  # percentage of fittest individuals to be kept as parents for next generation (elitist selection)

    # target concept value
    target_val = 0.9
    
    results, results_pop, results_gen = ALGE_class.what_if(n_runs, pop_size, generation, mutate_prob, retain, target_val)

    # visualize the results
    ALGE_class.visualize(results, results_pop, results_gen, target_val)