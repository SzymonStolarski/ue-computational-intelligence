import random
from math import sqrt


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
            rnd_coord_x = random.randint(self.COORDINATES_RANGE[0],
                                         self.COORDINATES_RANGE[1])
            rnd_coord_y = random.randint(self.COORDINATES_RANGE[0],
                                         self.COORDINATES_RANGE[1])

            if random.getrandbits(1):
                rnd_demand = random.randint(self.DEMAND_RANGE[0],
                                            self.DEMAND_RANGE[1])
                rnd_supply = 0
            else:
                rnd_demand = 0
                rnd_supply = random.randint(self.SUPPLY_RANGE[0],
                                            self.SUPPLY_RANGE[1])

            self.__generated_points[i] = {'coords': (rnd_coord_x, rnd_coord_y),
                                          'demand': rnd_demand,
                                          'supply': rnd_supply}
        self.__generated_points[0]['demand'] = 0
        self.__generated_points[0]['supply'] = 0
        self.__distances = self.__calculate_distances(self.__generated_points)

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

    @property
    def generated_points(self):
        return self.__generated_points

    @property
    def distances(self):
        return self.__distances
