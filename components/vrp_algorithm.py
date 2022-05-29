from json import load
from math import prod
import random

import matplotlib.pyplot as plt
import pandas as pd

from components.points_generator import PointsGenerator
from components.ga_components.selection import *
from components.ga_components.crossover import *
from components.ga_components.mutation import *


class VRPAlgorithm:
    MAX_LOAD = 2000
    PRODUCTS = ['Tuna', 'Oranges', 'Uran']

    def __init__(self, population_size: int, n_iterations: int,
                 selector: BaseSelector, crossover: BaseCrossover,
                 mutator: BaseMutator, maximize: bool = False,
                 verbose: bool = True,
                 dem_task_init_load: int = 2000,
                 sup_task_init_load: int = 0) -> None:
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.maximize = maximize
        self.verbose = verbose
        self.dem_task_init_load = dem_task_init_load
        self.sup_task_init_load = sup_task_init_load

        self.__init_magazines = {'demand': {
                                        'Tuna': random.randint(0, 4),
                                        'Oranges': random.randint(0, 4),
                                        'Uran': random.randint(0, 4)
                                    },
                                 'supply': {
                                        'Tuna': random.randint(0, 4),
                                        'Oranges': random.randint(0, 4),
                                        'Uran': random.randint(0, 4)
                                    }}
        self.__cat_initialization = self.__easter_egg_cat_initialization()

    def learn(self, data: PointsGenerator):
        self.pg = data
        self.__iteration_results = {}
        self.__best_result_change = {}
        self.__best_paths = pd.DataFrame()
        print(f"Cat in tuna demand car: \
            {self.__cat_initialization['demand_tuna']}")
        print(f"Cat in tuna supply car: \
            {self.__cat_initialization['supply_tuna']}")
        iteration = 0
        population = self._create_base_population(data.supply_df,
                                                  data.demand_df)
        population_scores, population_routes = self._evaluate(population)

        if not self.maximize:
            self.__iteration_results[iteration] = min(
                list(population_scores.values()))
            self.__best_result_change[iteration] = min(
                list(self.__iteration_results.values()))
            best_individual = min(
                population_scores, key=population_scores.get)
            self.__best_paths = population_routes[
                population_routes[
                    'ind_index'] == best_individual][[
                        'demand_routes', 'supply_routes']]

        else:
            self.__iteration_results[iteration] = max(
                list(population_scores.values()))
            self.__best_result_change[iteration] = max(
                list(self.__iteration_results.values()))
            best_individual = max(
                population_scores, key=population_scores.get)
            self.__best_paths = population_routes[
                population_routes[
                    'ind_index'] == best_individual][[
                        'demand_routes', 'supply_routes']]

        if self.verbose:
            print('Starting algorithm...')
            print(f'Iteration: {iteration}. \
                Result: {self.__iteration_results[iteration]}')
        while not iteration == self.n_iterations:
            # Increment iteration here because iteration 0 was base population
            iteration += 1

            # Selection
            selected_population = self.selector.selection(population,
                                                          population_scores,
                                                          self.maximize)
            # Mutation
            population = self.mutator.mutate(selected_population)
            # Crossover
            population = self.crossover.crossover(population)
            # Evaluation
            population_scores, population_routes = self._evaluate(population)

            if not self.maximize:
                self.__iteration_results[iteration] = min(
                    list(population_scores.values()))
                self.__best_result_change[iteration] = min(
                    list(self.__iteration_results.values()))
                if self.__best_result_change[iteration] <\
                        self.best_result_change[iteration-1]:
                    best_individual = min(
                        population_scores, key=population_scores.get)
                    self.__best_paths = population_routes[
                        population_routes[
                            'ind_index'] == best_individual][[
                                'demand_routes', 'supply_routes']]
            else:
                self.__iteration_results[iteration] = max(
                    list(population_scores.values()))
                self.__best_result_change[iteration] = max(
                    list(self.__iteration_results.values()))
                if self.__best_result_change[iteration] >\
                        self.best_result_change[iteration-1]:
                    best_individual = max(
                        population_scores, key=population_scores.get)
                    self.__best_paths = population_routes[
                        population_routes[
                            'ind_index'] == best_individual][[
                                'demand_routes', 'supply_routes']]

            if self.verbose and iteration % 5 == 0:
                print(f'Iteration: {iteration}. \
                Best result: {min(list(self.__iteration_results.values()))}')

        if not self.maximize:
            self.__best_score = min(list(self.__iteration_results.values()))
        else:
            self.__best_score = max(list(self.__iteration_results.values()))

        if self.verbose:
            print('Algorithm finished.')

        return self

    def _create_base_population(self,
                                supply_df: pd.DataFrame,
                                demand_df: pd. DataFrame) -> dict:
        base_population = {}
        for i in range(0, self.population_size):
            list_to_shuffle = list(supply_df['id'])\
                + list(demand_df['id'])
            random.shuffle(list_to_shuffle)
            base_population[i] = list_to_shuffle

        return base_population

    def _evaluate(self, population: dict) -> tuple:
        population_scores = {}
        # I need somewhere to store the routes
        # for later use on visualizations :D
        population_routes = pd.DataFrame(
            columns=['ind_index', 'demand_routes', 'supply_routes'])

        for index, individual in population.items():
            population_scores[index] = self.__evaluate_demand_task(
                individual)[0] + self.__evaluate_supply_task(individual)[0]
            temp_df = pd.DataFrame(
                [[index, self.__evaluate_demand_task(individual)[1],
                  self.__evaluate_supply_task(individual)[1]]],
                columns=['ind_index', 'demand_routes', 'supply_routes'])

            population_routes = pd.concat([population_routes, temp_df])

        return population_scores, population_routes

    def __evaluate_demand_task(self, individual: list[int]) -> tuple:
        order = {i: j for j, i in enumerate(individual)}

        df_demand = self.pg.demand_df
        demand_distance = 0
        demand_routes = {}
        for product in VRPAlgorithm.PRODUCTS:
            df_product = df_demand[['id', product]].copy()
            df_product['rank'] = df_product['id'].map(order)
            df_product.sort_values(by='rank', inplace=True)
            df_product.drop('rank', axis=1, inplace=True)
            df_product = df_product[df_product[product] != 0]

            car_route = []
            car_route.append(self.__init_magazines['demand'][product])
            load_sum = self.dem_task_init_load
            # Special condition if the cat travels in the demand tuna car
            if product == 'Tuna' and self.__cat_initialization['demand_tuna']:
                for _, row in df_product.iterrows():
                    if (load_sum - row[product] - self.pg.distances[
                         (car_route[-1], row['id'])]) < 0:
                        closest_magazine = self.__get_closest_magazine(
                            row['id'])
                        car_route.append(closest_magazine)
                        load_sum = VRPAlgorithm.MAX_LOAD
                    car_route.append(row['id'])
                    load_sum -= (self.pg.distances[(car_route[-1], row['id'])]
                                 + row[product])
            else:
                for _, row in df_product.iterrows():
                    if load_sum - row[product] < 0:
                        closest_magazine = self.__get_closest_magazine(
                            row['id'])
                        car_route.append(closest_magazine)
                        load_sum = VRPAlgorithm.MAX_LOAD
                    car_route.append(row['id'])
                    load_sum -= row[product]
            demand_routes[product] = car_route
            demand_distance += self.__calculate_route_distance(car_route)

        return demand_distance, demand_routes

    def __evaluate_supply_task(self, individual: list[int]) -> tuple:
        order = {i: j for j, i in enumerate(individual)}

        df_supply = self.pg.supply_df
        supply_distance = 0
        supply_routes = {}
        for product in VRPAlgorithm.PRODUCTS:
            df_product = df_supply[['id', product]].copy()
            df_product['rank'] = df_product['id'].map(order)
            df_product.sort_values(by='rank', inplace=True)
            df_product.drop('rank', axis=1, inplace=True)
            df_product = df_product[df_product[product] != 0]

            car_route = []
            car_route.append(self.__init_magazines['supply'][product])
            load_sum = self.sup_task_init_load
            # Special condition if the cat travels in the supply tuna car
            if product == 'Tuna' and self.__cat_initialization['supply_tuna']:
                for _, row in df_product.iterrows():
                    if (load_sum
                        - self.pg.distances[(car_route[-1], row['id'])]
                            + row[product]) > VRPAlgorithm.MAX_LOAD:
                        closest_magazine = self.__get_closest_magazine(
                            row['id'])
                        car_route.append(closest_magazine)
                        load_sum = 0
                    car_route.append(row['id'])
                    # A little simplification - the cats eats only if
                    # the car contains at least the same amount of tuna
                    if load_sum >= self.pg.distances[(car_route[-1],
                                                      row['id'])]:
                        load_sum -= self.pg.distances[(car_route[-1],
                                                       row['id'])]
                    load_sum += row[product]
            else:
                for _, row in df_product.iterrows():
                    if load_sum + row[product] > VRPAlgorithm.MAX_LOAD:
                        closest_magazine = self.__get_closest_magazine(
                            row['id'])
                        car_route.append(closest_magazine)
                        load_sum = 0
                    car_route.append(row['id'])
                    load_sum += row[product]
            supply_routes[product] = car_route
            supply_distance += self.__calculate_route_distance(car_route)

        return supply_distance, supply_routes

    def __get_closest_magazine(self, point: int) -> int:
        distance_dict = {}
        for i in self.pg.magazines_points:
            distance_dict[i] = self.pg.distances[(point, i)]

        return min(distance_dict, key=distance_dict.get)

    def __calculate_route_distance(self, route: list[int]) -> float:
        distance = 0
        for idx, i in enumerate(route):
            if idx == 0:
                previous_city = i
                continue
            distance += self.pg.distances[(previous_city, i)]
            previous_city = i

        return distance

    def __easter_egg_cat_initialization(self) -> dict:
        cat_initialization = {
            'demand_tuna': False,
            'supply_tuna': False
        }

        # First draw the product
        if (random.randint(0, len(VRPAlgorithm.PRODUCTS)-1)
           == VRPAlgorithm.PRODUCTS.index('Tuna')):
            # If Tuna has been selected, then select on which car
            # the cat drives (supply or demand)
            cat_initialization[
                random.choice(list(cat_initialization.keys()))] = True

        return cat_initialization

    @property
    def learning_visualization(self):
        plt.plot(self.__best_result_change.keys(),
                 self.__best_result_change.values(), color='green')
        plt.title('Learning visualization')
        plt.ylabel('objective function value')
        plt.xlabel('iteration')
        plt.show()

    @property
    def init_magazines(self):
        return self.__init_magazines

    @property
    def results(self):
        return self.__iteration_results

    @property
    def best_result_change(self):
        return self.__best_result_change

    @property
    def best_score(self):
        return self.__best_score

    @property
    def best_paths(self):
        return self.__best_paths

    def ile_janusz_zaoszczedzil(self) -> None:
        start_km = self.__iteration_results[0]
        end_km = self.__best_result_change[
            len(self.__best_result_change)-1]

        print(f"Gdyby Janusz na łoko (na łoko to jeden umar)\n\
            wyznaczał trasy swoich ciężarówek, to by zrobiły\n\
            one {round(start_km, 2)} km. Czyli przy łobecnych cenach dizla,\n\
            śr. 7.50 zł i średnim spalaniu dostawczaka 13l.,\n\
            to by wydoł {round(start_km*6.5, 2)} zł na paliwo. Kurła!")

        print(f"Gdyby Janusz się nos posłuchoł i kupił nosz pakiet\n\
            maszin lerningowy, to by jego ciężarówki zrobiły\n\
            {round(end_km, 2)} km, czyli o {round(start_km-end_km, 2)} km\n\
            mniej! Janusz by załoszczędził\n\
            {round(start_km*7.5-end_km*7.5, 2)} zł! Kurła!")

    # TODO: add property:
    # @property
    # def visualize_routes(self):
    #   # Implement logic here...
