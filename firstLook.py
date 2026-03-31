import matplotlib.pyplot as plt
import pandas as pd

# Path Constants
DATA_DIR = "data/data.csv"
metadata = "data/features_metadata.csv"

# Data Loading
df_data = pd.read_csv(DATA_DIR)
df_meta = pd.read_csv(metadata)

df_data.head()
df_meta.head()

lenOfData = []
for i in range (10):
  dataUnderI = df_data.columns[df_data.isnull().mean() < i/10]
  lenOfData.append(len(dataUnderI))
  print(f"Number of features with less than {i*10}% missing data: {len(dataUnderI)}")

#graph of the number of features according to the percentage of missing data
plt.figure(figsize=(10, 6))
plt.plot(range(10), lenOfData, marker='o')
plt.grid()
plt.title('Number of features according to the percentage of missing data')
plt.xlabel('Percentage of missing data (%)')
plt.ylabel('Number of features')
plt.xticks(range(10), [f'{i*10}%' for i in range(10)])
#save the graph as an image
plt.savefig('visualization/missing_data.png')
plt.show()