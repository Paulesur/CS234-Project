from GenerateEVs import chooseRandomCapacity, chooseRandomSOCInit, chooseRandomSOC_f


class EV:
    """
    Implement the EV class:
    - name: string
    - capacity: float
    - soc: float

    Methods:
    - updateSOC(P): update the state of charge of the EV
    """

    def __init__(self):
        self.capacity = chooseRandomCapacity()
        self.soc = chooseRandomSOCInit()
        self.soc_f = chooseRandomSOC_f(self.soc)
        self.soc_i = self.soc

    def updateSOC(self, P):
        self.soc = max(0, min(1, self.soc + P / (4 * self.capacity)))  # 15 minutes

    def reset(self):
        self.soc = 0
        self.soc_i = 0
        self.soc_f = 0

    def copy(self):
        e = EV()
        e.capacity = self.capacity
        e.soc = self.soc
        e.soc_f = self.soc_f
        e.soc_i = self.soc_i
        return e
