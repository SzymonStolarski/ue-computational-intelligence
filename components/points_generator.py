import random
from math import sqrt

from models.RoutePoint import RoutePoint


class PointsGenerator:
    COORDINATES_RANGE = (100, 200)
    DEMAND_RANGE = (100, 200)
    SUPPLY_RANGE = (100, 200)

    def __init__(self, n_points: int) -> None:
        self.__n_points = n_points
        self.__generated_points = list()
        self.__distances = dict()

    def generate_points(self):
        for i in range(0, self.__n_points):
            rndCoordX = random.randint(self.COORDINATES_RANGE[0], self.COORDINATES_RANGE[1])
            rndCoordY = random.randint(self.COORDINATES_RANGE[0], self.COORDINATES_RANGE[1])
            rndDemand = random.randint(self.DEMAND_RANGE[0], self.DEMAND_RANGE[1])
            rndSupply = random.randint(self.SUPPLY_RANGE[0], self.SUPPLY_RANGE[1])

            self.__generated_points.append(RoutePoint((rndCoordX, rndCoordY), rndDemand, rndSupply))

        self.__distances = self.__calculate_distance(self.__generated_points)

        return self
        pass

    def __calculate_distance(self, points: list) -> dict:
        distances = dict()
        for i, point_x in enumerate(points):
            for j, point_y in enumerate(points):
                distance = 0.0
                if i != j:
                    distance = sqrt(
                        pow((point_x.coords[0] - point_y.coords[0]), 2)
                        + pow((point_x.coords[1] - point_y.coords[1]), 2))
                distances[(i, j)] = distance

        return distances
        pass
