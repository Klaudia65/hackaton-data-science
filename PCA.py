import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_imputed = pd.read_csv('data/imputed_sample.csv')

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed)


#    - Garder 95% de la variance expliquée (ou choisir un nombre fixe de composantes)
pca = PCA(n_components=0.95)  # ou PCA(n_components=100) pour un nombre fixe
X_pca = pca.fit_transform(X_scaled)


print(f"Nombre de composantes principales retenues : {pca.n_components_}")
print(f"Variance expliquée par composante : {pca.explained_variance_ratio_}")
print(f"Variance cumulée expliquée : {pca.explained_variance_ratio_.cumsum()}")

# la variance expliquée (optionnel)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(),
         marker='o', linestyle='--')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée expliquée')
plt.title('Variance expliquée par les composantes principales')
plt.savefig('visualization/pca_variance_explained.png')
plt.grid()
plt.show()

# sauvegarder les données réduites
X_pca.to_csv('data/imputed_sample_pca.csv', index=False)