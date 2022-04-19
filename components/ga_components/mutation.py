from abc import ABC, abstractmethod
import random


class BaseMutator(ABC):

    @abstractmethod
    def mutate(self, population: dict) -> dict:
        return


class SwapMutator(BaseMutator):

    def __init__(self, mutation_probability: float) -> None:
        self.mutation_probability = mutation_probability

    def mutate(self, population: dict) -> dict:

        mutated_population = {}
        for idx, old_value in population.items():

            # Perform mutation based on probability
            random_number = random.random()
            if random_number <= self.mutation_probability:
                new_value = old_value.copy()

                swap_idx_1 = 0
                swap_idx_2 = 0
                while swap_idx_1 == swap_idx_2:
                    swap_idx_1 = random.randint(0, len(old_value)-1)
                    swap_idx_2 = random.randint(0, len(old_value)-1)

                new_value[swap_idx_1] = old_value[swap_idx_2]
                new_value[swap_idx_2] = old_value[swap_idx_1]

                mutated_population[idx] = new_value
            else:
                mutated_population[idx] = old_value

        return mutated_population


class InverseMutator(BaseMutator):

    def __init__(self, mutation_probability: float) -> None:
        self.mutation_probability = mutation_probability

    def mutate(self, population: dict) -> dict:

        mutated_population = {}
        for idx, old_value in population.items():

            # Perform mutation based on probability
            random_number = random.random()
            if random_number <= self.mutation_probability:
                new_value = old_value.copy()

                start_idx = 0
                end_idx = 0
                while (start_idx == end_idx):
                    while end_idx-start_idx <= 1:
                        start_idx = random.randint(0, len(old_value))
                        end_idx = random.randint(0, len(old_value))
                        if start_idx > end_idx:
                            start_idx, end_idx = end_idx, start_idx

                part_to_inverse = old_value[start_idx:end_idx]
                part_to_inverse.reverse()
                new_value[start_idx:end_idx] = part_to_inverse

                mutated_population[idx] = new_value
            else:
                mutated_population[idx] = old_value

        return mutated_population
