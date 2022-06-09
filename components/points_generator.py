import random
from math import sqrt

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd


class PointsGenerator:
    """Class to randomly generate points.

    The points generated with this class are used for the VRP algorithm.
    Ranges for both the coordinates and the amount of product supply/demand
    for given customer are defined in the class' constants space.

    Parameteres
    -----------
    n_points : int
        Number of points to be generated.
    """
    COORDINATES_RANGE = (0, 100)
    DEMAND_RANGE = (100, 200)
    SUPPLY_RANGE = (100, 200)

    def __init__(self, n_points: int) -> None:
        self.__n_points = n_points
        self.__generated_points = {}
        self.__distances = {}

    def generate_points(self):
        """Method that starts the generation of points.

        Returns
        -------
        self
            An instance of self.
        """
        for i in range(0, self.__n_points):
            rnd_coord_x = random.randint(PointsGenerator.COORDINATES_RANGE[0],
                                         PointsGenerator.COORDINATES_RANGE[1])
            rnd_coord_y = random.randint(PointsGenerator.COORDINATES_RANGE[0],
                                         PointsGenerator.COORDINATES_RANGE[1])

            # If true then demand - we deliver there
            # Else - this is a point from which we pickup
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
        """
        Method to calculate the euclidean distances between generated
        points.

        Parameters
        ----------
        points : dict
            Generated dictionary of points with their coordinates.

        Returns
        -------
        distances : dict
            Dictionary with distances between pairs of points.
        """
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

    def __create_supply_demand_dfs(self) -> tuple:
        """
        Create the dataframes with the supply or demand for each of the
        products for each customer.

        Notes
        -----
        The method generates a tuple of two ``pd.DataFrames``:
            - one for demand,
            - second for supply.
        Each of the dataframes contains columns with information with
        amount of the product to be shipped to/from.

        Returns
        -------
        tuple
            ``pd.DataFrames`` with demand and supply.
        """
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
        for _ in demand_points_ids:
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

    def visualize_points(self):
        """
        Method to visualize the generated points with distinction to
        customers and magazines.
        """
        customers = {i: j for i, j in zip(
            list(self.__generated_points.keys()),
            [i['coords'] for i in list(self.__generated_points.values())])
            if i not in self.__magazines_points}
        magazines = {i: j for i, j in zip(
            list(self.__generated_points.keys()),
            [i['coords']for i in list(self.__generated_points.values())])
            if i in self.__magazines_points}

        _, ax = plt.subplots(figsize=(15, 10), dpi=100)
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.title('Generated points', fontsize=20)
        ax.scatter(list(i[0] for i in list(customers.values())),
                   list(i[1] for i in list(customers.values())),
                   s=250, edgecolors='blue', color='white', linewidths=2)
        ax.scatter(list(i[0] for i in list(magazines.values())),
                   list(i[1] for i in list(magazines.values())),
                   s=250, edgecolors='red', color='white', linewidths=2)

        for i, c in self.__generated_points.items():
            ax.annotate(str(i), xy=c['coords'], fontsize=10, ha="center",
                        va="center", color="black")

        legend_elements = [Line2D([0], [0], marker='o', color='w',
                           markeredgecolor='blue', label='Customer',
                           markersize=15),
                           Line2D([0], [0], marker='o', color='w',
                           markeredgecolor='red', label='Magazine',
                           markersize=15)]
        ax.legend(handles=legend_elements, prop={'size': 13}, loc='best')

        plt.show()
