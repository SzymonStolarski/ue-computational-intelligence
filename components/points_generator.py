import random
from math import sqrt

import pandas as pd


class PointsGenerator:
    COORDINATES_RANGE = (100, 200)
    DEMAND_RANGE = (100, 200)
    SUPPLY_RANGE = (100, 200)

    def __init__(self, n_points: int) -> None:
        self.__n_points = n_points
        self.__generated_points = {}
        self.__distances = {}

    def generate_points(self):
        for i in range(0, self.__n_points):
            rnd_coord_x = random.randint(PointsGenerator.COORDINATES_RANGE[0],
                                         PointsGenerator.COORDINATES_RANGE[1])
            rnd_coord_y = random.randint(PointsGenerator.COORDINATES_RANGE[0],
                                         PointsGenerator.COORDINATES_RANGE[1])

            # If true then demand - we pickup from that point
            # Else - this is a point from which we
            if random.getrandbits(1):
                point_type = 'demand'
            else:
                point_type = 'supply'

            self.__generated_points[i] = {'coords': (rnd_coord_x, rnd_coord_y),
                                          'point_type': point_type}

        # Now we need to randomly select 5 points to be our magazines
        magazines_points = random.sample(range(0, self.__n_points), 5)
        for i in range(0, self.__n_points):
            if i in magazines_points:
                self.__generated_points[i]['point_type'] = 'magazine'

        self.__distances = self.__calculate_distances(self.__generated_points)
        self.__demand_df,\
            self.__supply_df = self.__create_supply_demand_dfs()
        self.__magazines_points = magazines_points

        return self

    def __calculate_distances(self, points: dict) -> dict:

        dict_of_coords = {}
        for k, val in points.items():
            dict_of_coords[k] = val['coords']

        distances = {}
        for i, _ in enumerate(dict_of_coords):
            for j, _ in enumerate(dict_of_coords):
                distance = sqrt(
                    pow((dict_of_coords[i][0] - dict_of_coords[j][0]), 2)
                    + pow((dict_of_coords[i][1] - dict_of_coords[j][1]), 2))

                distances[(i, j)] = distance

        return distances

    def __create_supply_demand_dfs(self):
        demand_points_ids = [x for x in self.__generated_points.keys()
                             if self.__generated_points[
                                 x]['point_type'] == 'demand']
        supply_points_ids = [x for x in self.__generated_points.keys()
                             if self.__generated_points[
                                 x]['point_type'] == 'supply']

        # Available products: 'Tuna', 'Uran', 'Oranges'
        # Demand
        demand_dict = {
            'Tuna': [],
            'Oranges': [],
            'Uran': []
        }
        list_of_products = list(demand_dict.keys())
        supply_dict = demand_dict.copy()
        supply_dict = {
            'Tuna': [],
            'Oranges': [],
            'Uran': []
        }
        for i in demand_points_ids:
            rnd_demand = random.randint(PointsGenerator.DEMAND_RANGE[0],
                                        PointsGenerator.DEMAND_RANGE[1])
            random.shuffle(list_of_products)
            for product in list_of_products:
                product_qty = random.randint(
                    0,
                    rnd_demand
                )
                demand_dict[product].append(product_qty)
                rnd_demand -= product_qty
        demand_dict['id'] = demand_points_ids

        # Supply
        for _ in range(len(supply_points_ids)):
            rnd_supply = random.randint(PointsGenerator.SUPPLY_RANGE[0],
                                        PointsGenerator.SUPPLY_RANGE[1])
            random.shuffle(list_of_products)
            for product in list_of_products:
                product_qty = random.randint(
                    0,
                    rnd_supply
                )
                supply_dict[product].append(product_qty)
                rnd_supply -= product_qty
        supply_dict['id'] = supply_points_ids

        # Convert dicts to pandas and change order of cols
        demand_df = pd.DataFrame(demand_dict)
        demand_df = demand_df[['id', 'Tuna', 'Oranges', 'Uran']]
        supply_df = pd.DataFrame(supply_dict)
        supply_df = supply_df[['id', 'Tuna', 'Oranges', 'Uran']]

        return demand_df, supply_df

    @property
    def generated_points(self):
        return self.__generated_points

    @property
    def distances(self):
        return self.__distances

    @property
    def demand_df(self):
        return self.__demand_df

    @property
    def supply_df(self):
        return self.__supply_df

    @property
    def magazines_points(self):
        return self.__magazines_points
