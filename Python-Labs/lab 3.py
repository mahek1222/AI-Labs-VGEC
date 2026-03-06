# #1. applied condition statement
climate_data = [{"city":"Ahmedabad","temperature":29,"carbon_footprint":500.57},
                {"city":"kalol","temperature":26,"carbon_footprint":400},
                {"city":"Gandhinagar","temperature":25,"carbon_footprint":200.40},
                {"city":"Rajkot","temperature":27,"carbon_footprint":400.67}]
high_temp_threshold= 26
high_temp_cities =[city for city in climate_data if city["temperature"]>high_temp_threshold]
print("Cities with high temperature(26 <)")
for  city in high_temp_cities:
    print(f"{city['city']}- {city['temperature']}")

# #2. calculate carbon using loops
total_carbon = 0
for city in climate_data:
    total_carbon += city["carbon_footprint"]
    avrg_footprint = total_carbon/len(climate_data)
    print("average carbon footprint is :",avrg_footprint)

#3.filter and manupalate datato find sustainable
sustainable_threshold=400
sustainable_city = list(filter(lambda city :city["carbon_footprint"]< sustainable_threshold,climate_data))
print("\n sustainable cities (carbon_footpront < 400 kg co2)")
for city in sustainable_city:
    print(f"{city['city']}-{city['carbon_footprint']} kg co2")

#4.analyze data to find city with carbon footprint
high_footprint_city= max(climate_data,key=lambda city:city["carbon_footprint"]) 
print(f"\ncity with th highest footprint :")
print(f"{high_footprint_city['city'] }- {high_footprint_city['carbon_footprint']}")