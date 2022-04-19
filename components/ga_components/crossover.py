from abc import ABC, abstractmethod
from audioop import cross
import random

import numpy as np


class BaseCrossover(ABC):

    @abstractmethod
    def crossover(self, population: dict) -> dict:
        return


class PMXCrossover(BaseCrossover):

    def __init__(self, crossover_probability: float) -> None:
        self.crossover_probability = crossover_probability

    def crossover(self, population: dict) -> dict:
        population_len = len(population)
        if population_len == 2:
            population_len += 2

        new_index = 0
        new_population = {}
        for i in range(2, population_len+2, 2):
            if i == 2:
                selected_pair = list(population.values())[:2]
            else:
                selected_pair = list(population.values())[i-2:i]

            parent_1 = selected_pair[0]
            parent_2 = selected_pair[1]

            random_number = random.random()
            if random_number <= self.crossover_probability:
                start_point, end_point = self.__crossover_points_generator(
                                        len(parent_1))
                core_parent_1 = parent_1[start_point:end_point]
                core_parent_2 = parent_2[start_point:end_point]
                mapping_table_1 = []
                # Creation add mappings to the mapping table
                # Also reverse the order for the second mapping table
                for index in range(len(core_parent_1)):
                    mapping_table_1.append([core_parent_2[index],
                                            core_parent_1[index]])
                mapping_table_2 = [[el[1], el[0]] for el in mapping_table_1]

                child_1 = self.__pmx_algorithm(parent_1, start_point,
                                               end_point, core_parent_2,
                                               mapping_table_1)
                child_2 = self.__pmx_algorithm(parent_2, start_point,
                                               end_point, core_parent_1,
                                               mapping_table_2)

                new_population[new_index] = child_1
                new_population[new_index+1] = child_2
            else:
                new_population[new_index] = parent_1
                new_population[new_index+1] = parent_2

            new_index += 2

        return new_population

    def __crossover_points_generator(self, length_of_individual: int) -> tuple:
        start_point = random.randint(1, length_of_individual-1)
        end_point = random.randint(1, length_of_individual-1)

        if start_point > end_point:
            start_point, end_point = end_point, start_point
        if start_point == end_point:
            start_point -= 1
            if start_point == 0:
                start_point += 1
                end_point += 1

        return start_point, end_point

    def __pmx_algorithm(self, parent: list, start_point: int, end_point: int,
                        core_of_second_parent: list, mapping_table: list):

        prefix = parent[:start_point]
        suffix = parent[end_point:]
        child = prefix + core_of_second_parent + suffix

        # Prefix Sequencing
        for index, value in enumerate(prefix):
            check = 0
            for pair in mapping_table:
                if value == pair[0]:
                    child[index] = pair[1]
                    check = 1
                    break
            if check == 0:
                child[index] = value

        # Suffix Sequencing
        for index, value in enumerate(suffix):
            check = 0
            for pair in mapping_table:
                if value == pair[0]:
                    child[index + end_point] = pair[1]
                    check = 1
                    break
            if check == 0:
                child[index + end_point] = value

        # Checks what unique numbers are used within the sequence
        unique_numbers = np.unique(child)

        # Check for when all numbers are used in sequence
        # Once condition is met recursion stops
        if len(child) > len(unique_numbers):
            child = self.__pmx_algorithm(child, start_point, end_point,
                                         core_of_second_parent, mapping_table)

        return child
