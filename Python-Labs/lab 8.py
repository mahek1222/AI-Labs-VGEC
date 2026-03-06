#handling data with pandas 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_AI_tools={
    "tool_name":["chatGpt","lovable","customGPT","gamma"],
    "carbon_emmition":[0.3,0.6,np.nan,0.8],
    "cost":[10,20,15,18]
}
tool_df=pd.DataFrame(data_AI_tools)
# print("data of AI tools which emit how much carbon")
# print(tool_df)
remove_df = tool_df.dropna()
# print(remove_df)
#fill missing data
tool_df["carbon_emmition"]=tool_df["carbon_emmition"].fillna(tool_df["carbon_emmition"].mean())
print(tool_df)
tool_df["cost"]=tool_df["cost"].fillna("premium")
print(tool_df)
#create a simple plot
sns.pairplot(tool_df)
plt.show()


