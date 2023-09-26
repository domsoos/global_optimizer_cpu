#include "utility.h"
#include "genetic.h"

// Generate initial population
std::vector<Individual> init_population(std::function<double(std::vector<double> &)> func, int dim, std::vector<double> x0, int pop_size, bool use_bfgs) {
    std::vector<Individual> population;

    for (int i = 0; i < pop_size; i++) {
        Individual ind;
        //ind.genes = x0;
        std::vector<double> genes;
        for (int i=0; i<=dim; i++){
            double random = static_cast<double>(rand(-10.5, 10.5));
            genes.push_back(random);
        }//end for
        ind.genes = genes;
        ind.fitness = optimize(func, ind.genes, use_bfgs, 1e-12, 2500); 
        population.push_back(ind);
    }
    return population;
}// end init_population


// Selection based on fitness 
Individual tournament_selection(std::vector<Individual> population) {
    // Pick random individuals, avoid picking last index
    Individual ind1 = population[rand(0, population.size()-2)]; 
    Individual ind2 = population[rand(0, population.size()-2)];

    // Return the better individual
    if (ind1.fitness < ind2.fitness) {
        return ind1;
    } else {  
        return ind2;
    }// end if-else block
}// end tournament_selection

// Crossover
std::vector<Individual> crossover(std::function<double(std::vector<double> &)> func, Individual ind1, Individual ind2, bool use_bfgs) {
    std::vector<Individual> offspring;
    offspring.resize(2);

    // Take random mix of genes from parents
    // Take first gene from ind1 and second from ind2 for offspring1
    offspring[0].genes = {ind1.genes[0], ind2.genes[1]};

    // Take first gene from ind2 and second from ind2 for offspring2
    offspring[1].genes = {ind2.genes[0], ind1.genes[1]};

    // Evaluate the fitness of each offspring from the parents
    offspring[0].fitness = optimize(func, offspring[0].genes,use_bfgs, 1e-12, 500);
    offspring[1].fitness = optimize(func, offspring[1].genes,use_bfgs, 1e-12, 500);

    return offspring;
}// end crossover

// Mutation function
void mutate(Individual &ind) {
    // with 15% probability
    if (rand(0, 1) < 0.15) {
        // mutate one of the genes by adding a small random value between -0.25 and 0.25
        ind.genes[rand(0, ind.genes.size() - 1)] += rand(-0.25, 0.25); 
    }// end if
}// end mutate

// Genetic algorithm 
std::vector<Individual> genetic_algo(std::function<double(std::vector<double> &)> func, int max_gens, int pop_size, int dim, std::vector<double> x0, bool use_bfgs) {
    std::vector<Individual> population = init_population(func, dim, x0,pop_size, use_bfgs);

    for (int gen = 0; gen < max_gens; gen++) {

        // Create next generation
        std::vector<Individual> next_gen;
        // create next generation based on the mutation of the previous one    
        for (int i = 0; i < pop_size; i++) { 
            
            // Selection
            Individual ind1 = tournament_selection(population);
            Individual ind2 = tournament_selection(population);

            // Crossover
            auto offspring = crossover(func, ind1, ind2, use_bfgs);

            // Mutation
            mutate(offspring[0]);
            mutate(offspring[1]);

            // Add to next generation
            if(offspring[0].fitness < offspring[1].fitness) {
                next_gen.push_back(offspring[0]);
            } else {
                next_gen.push_back(offspring[1]);
            }
        }// end pop_size for

        // Find the global minimum in the current population
        for (auto& individual : next_gen) {
            if (individual.fitness < global_min) {
                std::cout<<"\n\ngenetic New Global Minimum: " << individual.fitness << " w/ params: \n";
                global_min = individual.fitness; 
                best_params = {};
                for(int i=0;i<individual.genes.size();i++) {
                    best_params.push_back(individual.genes[i]);
                    std::cout <<"x["<<i<<"]: " << individual.genes[i]<<std::endl;
                }
            }// end if


        }//end for

        // Replace population for next generation
        if (next_gen.size() == pop_size) {
            population = next_gen;
        } else {
            std::cout << "Error: next_gen size mismatch" << std::endl;  
            std::cout << "Current population = " << population.size() << std::endl;
            std::cout << "Next generation = " << next_gen.size() << std::endl;
        }// end if-else
    }// end main for
    return population;
}// end genetic_algo