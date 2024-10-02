import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from minepy import MINE
import time

df = pd.read_excel('MIC.xlsx')

mine = MINE()

columns = df.columns
mic_matrix = pd.DataFrame(index=columns, columns=columns)
print(columns)

start_time=time.time()

for i in columns:
    for j in columns:
        mine.compute_score(df[i], df[j])
        mic_matrix.loc[i, j] = mine.mic()

end_time=time.time()

run_time=end_time-start_time
print(run_time)

row_indices = ['Data','IMF1', 'IMF2', 'IMF3','IMF4', 'IMF5', 'IMF6','IMF7', 'IMF8', 'IMF9','IMF10','IMF11','IMF12','IMF13','IMF14','IMF15']
col_indices = ['Data','EF1', 'EF2', 'EF3','EF4','Efa','Efj','EF5','EF6','EF7','EF8','EF9','EF10','EF11','EF12','EF13','EF14','EF15','EF16','EF17','EF18', 'EF19', 'EF20','EF21','EF22','EF23']

mic_matrix_filtered = mic_matrix.loc[row_indices, col_indices]

plt.figure(figsize=(20, 8))
sns.heatmap(mic_matrix_filtered.astype(float), annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.title('MIC Heatmap')
plt.show()
