#create function to calculate carbon footprint
def cal_carbon_footprint(enrgy_consumption,emmition_factor):
    return enrgy_consumption*emmition_factor

#implimentation of function
enrgy_consumption= 1000
emmition_factor= 4.32
carbon_footprint=cal_carbon_footprint(enrgy_consumption,emmition_factor)
print(f"carbon footprint: {carbon_footprint} kg co2")