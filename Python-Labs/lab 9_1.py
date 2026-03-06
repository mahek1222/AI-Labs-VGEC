import matplotlib.pyplot as plt

months = ['January','February','March','April','May','June']
Energy_Consumption=[1200,1300,1100,1500,1400,1600]

plt.plot(months, Energy_Consumption, marker ='o', color='b', linestyle= '--')

plt.title('Energy consumption Over 6 Months')
plt.xlabel('Month ')
plt.ylabel('Energy consumption (MWh)')
plt.show()

Energy_source= ['solar','wind','Hydropower','Biomass']
Energy_values = [1200,34000,2900,2500]

plt.bar(Energy_source,Energy_values,color='green')
plt.title('Energy consumption by Renewable Energy Source')
plt.xlabel('Energy source')
plt.ylabel('Energy consumption (Mwh)' )
plt.show()

plt.pie(Energy_values, labels= Energy_source,autopct='%1.1f%%',colors=['green','yellow','blue','orange'])

plt.title('Energy consumption share by source')
plt.show()

carbon_emission= [200,500,450,300]

plt.scatter(Energy_values,carbon_emission,color='red')

plt.title('Energy consumption vs carbon emission')
plt.xlabel('Energy consumption (mwh)')
plt.ylabel('carbon emission (kg co2)')
plt.show()

plt.bar(Energy_source,Energy_values,color='orange',edgecolor ='black')
plt.title('customized energy consumtion by source',fontsize=14,fontweight= 0.5)
plt.xlabel('Energy source',frontsize=12)
plt.ylabel('Energy consumption (MWh)',fontsize=12)

plt.grid(True,linestyle='--',alpha=0.7)
plt.legend(['Energy consumption'], loc='upper left')
plt.show()
