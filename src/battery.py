# Battery class to simulate the behavior of a battery in terms of its capacity, state of charge (SOC), and power constraints.

# Attributes:
#     capacity (float): The total capacity of the battery in MWh.
#     soc (float): The state of charge of the battery, represented as a value between 0 and 1.
#     max_power (float): The maximum charge and discharge power of the battery in MW (absolute value).
#     DeltaCharge (float): The power increment of the battery, meaning that the power delivered by the battery will be a multiple of this value.

# Methods:
#     __init__(capacity, soc, max_power):
#         Initializes the Battery instance with the given capacity, state of charge, and maximum power.

#     reset():
#         Resets the state of charge of the battery to 0.

#     updateBattery(amount):
#         Updates the state of charge of the battery based on the given amount.
#         - amount > 0: charges the battery
#         - amount < 0: discharges the battery
#         Assumes that the amount is chosen carefully to respect the constraints.

#     copy():
#         Creates and returns a copy of the current Battery instance.


class Battery:
    def __init__(self, capacity, soc, max_power):
        self.capacity = capacity  # in MWh
        self.soc = soc  # between 0 and 1
        self.max_power = (
            max_power  # Max charge and discharge power in MW (in absolute value)
        )
        self.DeltaCharge = 0.1  # power increment of the battery, meaning that the power delivered by the battery will be a multiple of this value

    def reset(self):
        """
        Reset the state of charge of the battery
        """
        self.soc = 0

    def updateBattery(self, amount):
        """
        Update the state of charge of the battery
        ----
        - amount > 0: charge the battery
        - amount < 0: discharge the battery
        Here, we assume that the amount is already choosen carefullly to respect the constraints

        """
        self.soc = max(0, min(self.soc + 0.25 * amount / (self.capacity + 1e-10), 1))

    def copy(self):
        b = Battery(self.capacity, self.soc, self.max_power)
        return b
