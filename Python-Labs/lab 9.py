import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data ={
    "Solar": [1200,1500,1300],
    "wind" :[3400,3600,3200],
    "Hydropower":[2900,3100,2800],
    "biomass ": [2500,2700,2400]
}

df = pd.DataFrame(data)
sns.pairplot(df)
plt.show()

correlation_matrix=df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title ('correlation between energy source ')
plt.show()

sns.boxenplot(data=df)

plt.title('dstribution of energy consumption by source ')
plt.show()
sns.violineplot(data=df)

plt.title('violine plot of nergy consumption')
plt.show()

energy_values = [100, 200, 300, 400, 500]
carbon_emissions = [10, 20, 30, 40, 50]

df_reg =pd.DataFrame({
    'Energy Consumption ': energy_values,
    'carbon emission ': carbon_emissions
})

sns.regplot(x = 'Energy consumption ',y = 'carbom Emission',data=df_reg)

plt.title('Energy consumption vs carbon emissions ')
plt.xlabel('Energy consumption')
plt.ylabel('carbon emission')
plt.show()

months = ['January','February','March','April','May','June']

df_facet= pd.DataFrame({
    'month':month * 3,
    'Energy Consumption':[1200,1300,1100,1500,1400,1600] * 3,
    'Region ': ['Noth','south','East'] * 6
})

g = sns.FacetGrid(df_facet, col ='Region',hue='Region', height=4, aspect=1)
g.map(sns.lineplot,'month', 'Energy Consumption')
plt.subplots_adjust(top=0.85)
g.fig.suptitle('energy consumption over time by region')

plt.show()
