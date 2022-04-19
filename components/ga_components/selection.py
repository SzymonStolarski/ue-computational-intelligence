from abc import ABC, abstractmethod
import random


class BaseSelector(ABC):

    @abstractmethod
    def selection(self, population: dict, evaluated_population: dict,
                  maximize: bool = False) -> list:
        return

    def _select_population(self, population: dict,
                           selected_individuals_keys: list):
        """
        Function that based on previous population selects
        new population based on list of keys obtained from
        selection method.
        """

        selected_population_vals = [population[i] for i
                                    in selected_individuals_keys
                                    if i in population.keys()]
        selected_population = {k: v for k, v in zip(range(0,
                               len(selected_individuals_keys)),
                               selected_population_vals)}

        return selected_population


class ElitismSelector(BaseSelector):

    def __init__(self, percent: float) -> None:
        self.percent = percent

    def selection(self, population: dict, evaluated_population: dict,
                  maximize: bool = False) -> dict:

        no_of_elite_individuals = (int(len(
                                   evaluated_population)*self.percent))
        sorted_evaluation = {k: v for k, v in sorted(
                             evaluated_population.items(),
                             key=lambda item: item[1], reverse=maximize)}
        best_individuals = [k for k in
                            list(
                                 sorted_evaluation.keys()
                                 )[:no_of_elite_individuals]
                            ]

        missing_multiplier = int(
                            len(evaluated_population)/no_of_elite_individuals
                                )
        selected_individuals_keys = missing_multiplier*best_individuals
        if len(selected_individuals_keys) < len(evaluated_population):
            difference = (len(evaluated_population)-len(
                selected_individuals_keys))
            selected_individuals_keys.extend(
                selected_individuals_keys[:difference])

        selected_population = self._select_population(
            population, selected_individuals_keys)

        return selected_population


class TournamentSelector(BaseSelector):

    def __init__(self, k_percent: float) -> None:
        self.k_percent = k_percent

    def selection(self, population: dict, evaluated_population: dict,
                  maximize: bool = False) -> dict:

        population_size = len(evaluated_population)
        no_of_ind_in_tournament = int(self.k_percent*population_size)

        selected_individuals_keys = []
        iterator = 0
        while not iterator == population_size:
            # Select individuals keys that will participate in tournament
            selected_keys = random.sample(
                            range(population_size), no_of_ind_in_tournament
                            )
            # Create tournament dictionary
            tournament = {k: evaluated_population[k] for k in selected_keys}
            if not maximize:
                winner = min(tournament, key=tournament.get)
            else:
                winner = max(tournament, key=tournament.get)

            selected_individuals_keys.append(winner)
            iterator += 1

        selected_population = self._select_population(
            population, selected_individuals_keys)

        return selected_population


class RouletteSelector(BaseSelector):

    def __init__(self) -> None:
        pass

    def selection(self, population: dict, evaluated_population: dict,
                  maximize: bool = False) -> dict:

        sum_of_scores = sum([v for v in evaluated_population.values()])
        if not maximize:
            dict_of_share_initial = {k: v/sum_of_scores for k, v
                                     in evaluated_population.items()}
            dict_of_share_sorted = {k: v for k, v in sorted(
                                    dict_of_share_initial.items(),
                                    key=lambda item: item[1], reverse=False)}
            reversed_keys = list(dict_of_share_sorted.keys())
            reversed_keys.reverse()
            dict_of_share = {k: v for k, v in zip(reversed_keys,
                             dict_of_share_sorted.values())}
        else:
            dict_of_share = {k: v/sum_of_scores for k, v
                             in evaluated_population.items()}
            dict_of_share = {k: v for k, v in sorted(dict_of_share.items(),
                             key=lambda item: item[1], reverse=False)}

        ranges = []
        for idx, value in enumerate(dict_of_share.values()):
            if idx == 0:
                current_range = [0, value]
                previous_value = value
                ranges.append(current_range)
                continue

            current_range = [previous_value, previous_value+value]
            previous_value = previous_value+value
            ranges.append(current_range)
        dict_with_ranges = {k: v for k, v in zip(dict_of_share.keys(), ranges)}

        selected_individuals_keys = []
        iterator = 0
        while not iterator == len(evaluated_population):
            random_number = random.random()
            selected_individual = [k for k, v in dict_with_ranges.items()
                                   if random_number > v[0]
                                   and random_number <= v[1]
                                   ][0]
            selected_individuals_keys.append(selected_individual)
            iterator += 1

        selected_population = self._select_population(
            population, selected_individuals_keys)

        return selected_population
