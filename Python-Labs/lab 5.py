#create class for energy system and create method to calculate carbon footprit
class Energysystem:
    def __init__(self,building_name,energy_consumption,emmition_factor):
        self.building_name=building_name
        self.energy_consumption=energy_consumption
        self.emmition_factor=emmition_factor

    def _calculate_carbonfp(self):
        return self.energy_consumption*self.emmition_factor
    
    def _calculate_energy_saving(self):
        return self.energy_consumption* 0.10
building= Energysystem ("building c",499,23)
carbon_footprint=building._calculate_carbonfp()
saving_energy=building._calculate_energy_saving()

print(f"calculate carbon footprint is",carbon_footprint)
print(f"energy saving is",saving_energy)

#inheritance 
class SolarEnergy(Energysystem):
    def __init__(self,building_name,energy_consumption,emmition_factor,solar_production):
         super().__init__(building_name,energy_consumption,emmition_factor)
         self.solar_production = solar_production
    def net_energy_consumption(self):
        return self.energy_consumption - self.solar_production
solar_building = SolarEnergy("solar building ",500,0.15,1500)
net_consumption = solar_building.net_energy_consumption()
print(f"net consumption is",net_consumption)