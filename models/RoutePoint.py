class RoutePoint:

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = value

    @property
    def demand(self):
        return self._demand

    @demand.setter
    def demand(self, value):
        self._demand = value

    @property
    def supply(self):
        return self._supply

    @supply.setter
    def supply(self, value):
        self._supply = value

    def __init__(self, coords, demand, supply):
        self.coords = coords
        self.demand = demand
        self.supply = supply
