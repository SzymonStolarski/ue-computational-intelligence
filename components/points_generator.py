class PointsGenerator:
    COORDINATES_RANGE = (100, 200)
    DEMAND_RANGE = (100, 200)
    SUPPLY_RANGE = (100, 200)

    def __init__(self, n_points: int) -> None:
        self.__n_points = n_points

    def generate_points(self):

        # TODO: implement logic here
        # output for now should be a dict like:
        # self.__generated_points =
        # {0: {'coords': (12, 32), 'demand': 120, 'supply': 130},
        #  1: {'coords': (56, 11), 'demand': 100, 'supply': 100},
        #  ...
        #  n_points
        # }
        # self.__distances = self.__calculate_distance(
        #                           self.__generated_points)
        # return self
        pass

    def __calculate_distance(self, points: dict) -> dict:

        # TODO: implement logic to calculate the distances from
        # self.__generated_points[x]['coords']
        # output should look like:
        # {(0, 1): 32,
        #  (0, 2): 12,
        #  ...
        #  (1, 0): 32}
        pass
