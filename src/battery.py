class Battery:
    def __init__(self, capacity, soc, max_charge_power, max_discharge_power):
        self.capacity = capacity  # in MWh
        self.soc = soc  # between 0 and 1
        self.max_charge_power = max_charge_power  # in MW
        self.max_discharge_power = max_discharge_power  # in MW

    def updateBattery(self, amount):
        """
        Update the state of charge of the battery
        ----
        - amount > 0: charge the battery
        - amount < 0: discharge the battery
        Here, we assume that the amount is already choosen carefullly to respect the constraints

        """
        self.soc += 0.25 * amount / self.capacity
