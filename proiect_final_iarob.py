import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

#1. PARAMETRI (Configurarea Algoritmului)
POPULATION_SIZE = 60        # Dimensiunea populatiei 
GENES_PER_COORD = 10        # Numar biti per coordonata 
CHROMOSOME_LENGTH = GENES_PER_COORD * 2 
GENERATIONS = 50            # Numarul de epoci/generatii
MUTATION_RATE = 0.01        # Probabilitate mutatie (Pm ~ sub 1%)
INCEST_THRESHOLD = 5        # Distanta Hamming minima pentru Anti-Incest

X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100

# TINTA ALEATORIE (Explorare in mediu necunoscut)
TARGET_X = random.uniform(10, 90)
TARGET_Y = random.uniform(10, 90)

simulation_running = True

#2. FUNCTII MATEMATICE & REPREZENTARE

def binary_to_decimal(binary_list, min_val, max_val):
    """
    Transforma sirul binar in valoare reala
    Implementeaza formula de discretizare din curs:
    xi = ai + ValZec * (bi - ai) / (2^mi - 1)
    """
    decimal = 0
    for bit in binary_list:
        decimal = (decimal << 1) | bit
    
    # 2^mi - 1 (Valoarea maxima posibila cu acesti biti)
    max_dec = (2 ** len(binary_list)) - 1
    
    if max_dec == 0: return min_val
    # Aplicarea formulei de mapare pe intervalul [min, max]
    return min_val + (decimal / max_dec) * (max_val - min_val)

def decode_individual(chromosome):
    """ 
    Decodificare Genotip -> Fenotip
    Sparge cromozomul in coordonate X si Y 
    """
    genes_x = chromosome[:GENES_PER_COORD]
    genes_y = chromosome[GENES_PER_COORD:]
    x = binary_to_decimal(genes_x, X_MIN, X_MAX)
    y = binary_to_decimal(genes_y, Y_MIN, Y_MAX)
    return x, y

def calculate_fitness(chromosome):
    """ 
    Functia de supravietuire (Fitness)
    Relatia cu functia obiectiv (distanta): F(X) ~ 1/f(x)
    Cu cat distanta e mai mica, fitness-ul e mai mare
    """
    x, y = decode_individual(chromosome)
    distance = math.sqrt((x - TARGET_X)**2 + (y - TARGET_Y)**2)
    # Adaugam 0.001 pentru a evita impartirea la zero
    return 1 / (distance + 0.001)

def hamming_distance(ind1, ind2):
    """
    Calculeaza Distanta Hamming intre doi indivizi
    Metoda Impotriva Incestului
    """
    dist = 0
    for i in range(len(ind1)):
        if ind1[i] != ind2[i]:
            dist += 1
    return dist

#3. OPERATORI GENETICI

def selection_roulette(population, fitness_scores):
    """ 
    Selectie Proportionala (Tip Ruleta / Monte-Carlo)
    Sansa de selectie este proportionala cu fitness-ul
    """
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, individual in enumerate(population):
        current += fitness_scores[i]
        if current > pick:
            return individual
    return population[-1]

def crossover_uniform(parent1, parent2):
    """ 
    Incrucisare Uniforma
    Simuleaza o 'masca' generata aleator cu distributie uniforma
    Bitii sunt preluati de la parinti conform mastii (50% sansa)
    """
    child1, child2 = [], []
    for i in range(len(parent1)):
        # Generare bit masca (0 sau 1 cu prob 0.5)
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

def mutate(individual):
    """ 
    Mutatie
    Altereaza informatia genetica prin inversarea bitului
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

def on_close(event):
    global simulation_running
    print("\n[INFO] Fereastra a fost inchisa manual. Oprire simulare...")
    simulation_running = False

#4. MAIN

if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9)) 
    fig.canvas.mpl_connect('close_event', on_close)

    # Initializare populatie - Distributie Uniforma
    population = []
    for _ in range(POPULATION_SIZE):
        population.append([random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)])

    history_best_x = []
    history_best_y = []

    print(f"\n=== START SIMULARE (Algoritm Genetic cu Anti-Incest) ===")
    print(f"TINTA (Ascunsa): X={TARGET_X:.2f}, Y={TARGET_Y:.2f}")
    print("-" * 65)
    print(f"{'GEN':<5} | {'BEST ROBOT X':<12} | {'BEST ROBOT Y':<12} | {'DISTANTA':<10}")
    print("-" * 65)

    for gen in range(GENERATIONS):
        if not simulation_running:
            break

        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        # Gasim cel mai bun individ
        best_idx = np.argmax(fitness_scores)
        best_x, best_y = decode_individual(population[best_idx])
        
        dist_to_target = math.sqrt((best_x - TARGET_X)**2 + (best_y - TARGET_Y)**2)
        
        # AFISARE TABELAR
        print(f"{gen + 1:<5} | {best_x:<12.2f} | {best_y:<12.2f} | {dist_to_target:<10.2f}")
        
        history_best_x.append(best_x)
        history_best_y.append(best_y)
        
        # VIZUALIZARE
        ax.clear()
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_title(f"Gen {gen + 1} / {GENERATIONS} - Explorare\nDistanta: {dist_to_target:.2f}")
        
        # Desenare elemente
        ax.plot(TARGET_X, TARGET_Y, 'rX', markersize=15, label='Tinta')
        
        pop_x = [decode_individual(ind)[0] for ind in population]
        pop_y = [decode_individual(ind)[1] for ind in population]
        ax.scatter(pop_x, pop_y, c='blue', alpha=0.3, s=30, label='Populatie')
        
        ax.plot(history_best_x, history_best_y, 'g--', alpha=0.5, linewidth=1, label='Traseu Explorare')
        ax.plot(best_x, best_y, 'go', markersize=12, label='Best Robot')
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.draw()
        plt.pause(0.5) 
        
        # EVOLUTIA
        new_population = []
        new_population.append(population[best_idx])
        
        while len(new_population) < POPULATION_SIZE:
            # 1. Selectie
            p1 = selection_roulette(population, fitness_scores)
            p2 = selection_roulette(population, fitness_scores)
            
            # 2. Impotrivire la incest
            # Daca distanta Hamming e mica, parintii sunt prea similari, se cauta alt partener
            attempts = 0
            while hamming_distance(p1, p2) < INCEST_THRESHOLD and attempts < 10:
                p2 = selection_roulette(population, fitness_scores)
                attempts += 1
                
            # 3. Incrucisare Uniforma
            c1, c2 = crossover_uniform(p1, p2)
            
            # 4. Mutatie
            c1 = mutate(c1)
            c2 = mutate(c2)
            
            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)
                
        population = new_population

    print("-" * 65)
    print("Simulare finalizata.")
    plt.ioff()
    if simulation_running:
        plt.show()