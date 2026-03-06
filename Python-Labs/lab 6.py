#import pandas and create a databas
import pandas as pd
import numpy as np

renewable_energy = ["solar","wind","Hydropower y"]
data = {
    "project":["solar","wind","hydropower"],
    "capacity":[150,300,200],
    "cost":[200,400,350]
}
renewable_series = pd.Series(renewable_energy)
print("Renewable energy ")
print(renewable_series)

#creating datafram for green technology and shaw particular column
datafram= pd.DataFrame(data)
print("dataframe of green technology :")
print(data)
print("cost of green technology :")
print(data["cost"])

#filtering for capacity higher than 100 mw
capacity_np= np.array(data["capacity"])
high_capacity = capacity_np[capacity_np>100]
print("higheat capacity mw :")
print(high_capacity)
