import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_imputed = pd.read_csv('data/imputed_sample_linear.csv')

# 1st column is index, we need to separate it before PCA
index_col = df_imputed.iloc[:, 0]  # Remplacez par df_imputed['nom_de_la_colonne'] si elle a un nom
data = df_imputed.iloc[:, 1:]       # Données sans la première colonne

# normalisation (uniquement sur les données numériques)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 3. Appliquer la PCA
pca = PCA(n_components=0.95)  # ou PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

print(f"Nombre de composantes principales retenues : {pca.n_components_}")
print(f"Variance expliquée par composante : {pca.explained_variance_ratio_}")
print(f"Variance cumulée expliquée : {pca.explained_variance_ratio_.cumsum()}")

# Visualisation de la variance expliquée
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(),
         marker='o', linestyle='--')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée expliquée')
plt.title('Variance expliquée par les composantes principales')
plt.grid()
plt.savefig('visualization/pca_variance_explained_ridge.png')
plt.show()

#index column
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
df_pca.insert(0, index_col.name, index_col)  # Réintègre la colonne d'index

df_pca.to_csv('data/imputed_sample_linear_pca_with_index.csv', index=False)