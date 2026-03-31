import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)
    return 'f1', f1_score(y_true, y_hat), True

def cramers_v(confusion_matrix):
    """Calcule le V de Cramér."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()

    if n == 0:
        return 0

    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0


if __name__ == "__main__":
    data = pd.read_csv('data/data.csv')
    cols_to_drop = []
    for column in data.columns:
        rows = data[column].shape[0]
        non_missing = data[column].dropna().shape[0]
        missing = rows - non_missing
        missing_percentage = (missing / rows) * 100

        if missing_percentage > 60:
            cols_to_drop.append(column)

    data.drop(columns=cols_to_drop, inplace=True)

    data.info()

    add = pd.read_csv('data/ground_truth_train.csv')
    data = data.merge(add[['SEQN', 'MORTSTAT_2019']], on='SEQN', how='left')

    # Exclure la target et éventuellement SEQN
    vars_tab = [col for col in data.columns if col not in ["MORTSTAT_2019", "SEQN", "target"]]

    # Garder seulement les lignes qui ont une target
    data = data.dropna(subset=["MORTSTAT_2019"])

    X = data.drop(columns=["MORTSTAT_2019", "SEQN"], errors="ignore")
    y = data["MORTSTAT_2019"].astype(int)

    y_df = pd.read_csv("data/ground_truth_train.csv", index_col="SEQN")
    test_idx = pd.read_csv("data/test_indexes.csv", header=None)[0].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50)]
    )

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    print("F1 score:", f1)






    # # Heatmap des NaN
    # plt.figure(figsize=(14, 8))
    # sns.heatmap(data.isna(), cbar=False, yticklabels=False)
    # plt.title("Heatmap des valeurs manquantes")
    # plt.tight_layout()
    # plt.savefig("heatmap_missing.png")
    # plt.close()

    # # Heatmap corrélation avec la target
    # numeric_data = data.select_dtypes(include=['number'])

    # if "MORTSTAT_2019" in numeric_data.columns:
    #     corr_target = numeric_data.corr()[['MORTSTAT_2019']].sort_values(by='MORTSTAT_2019', ascending=False)

    #     plt.figure(figsize=(20, 30))
    #     sns.heatmap(corr_target, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    #     plt.title("Corrélation avec MORTSTAT_2019")
    #     plt.tight_layout()
    #     plt.savefig("heatmap_target.png")
    #     plt.close()


























































    # os.makedirs("crosstab", exist_ok=True)

    # significant_vars = []

    # for col in vars_tab:
    #     print(f"\n===== {col} =====")

    #     # On enlève les NaN pour le test
    #     subset = data[[col, "MORTSTAT_2019"]].dropna()

    #     # Éviter les colonnes vides ou constantes
    #     if subset.empty or subset[col].nunique() < 2:
    #         print(f"{col} ignorée : pas assez de modalités.")
    #         continue

    #     counts = pd.crosstab(subset[col], subset["MORTSTAT_2019"])

    #     # Vérifier qu'on a bien au moins 2 dimensions
    #     if counts.shape[0] < 2 or counts.shape[1] < 2:
    #         print(f"{col} ignorée : tableau insuffisant pour chi2.")
    #         continue

    #     perc = pd.crosstab(subset[col], subset["MORTSTAT_2019"], normalize="index") * 100

    #     print(counts)
    #     print("\n(%)")
    #     print(perc.round(1))

    #     # Test du chi²
    #     chi2, p_value, dof, expected = chi2_contingency(counts)

    #     # Taille d'effet
    #     v = cramers_v(counts)

    #     print(f"p-value = {p_value:.6f}")
    #     print(f"Cramér's V = {v:.4f}")

    #     # Critères de sauvegarde
    #     if p_value < 0.05 and v >= 0.1:
    #         significant_vars.append((col, p_value, v))

    #         plt.figure(figsize=(8, 5))

    #         if 1 in counts.columns:
    #             mort_vals = counts[1]
    #         else:
    #             mort_vals = pd.Series(0, index=counts.index)

    #         if 0 in counts.columns:
    #             alive_vals = counts[0]
    #         else:
    #             alive_vals = pd.Series(0, index=counts.index)

    #         plt.bar(counts.index.astype(str), mort_vals, label='mort')
    #         plt.bar(counts.index.astype(str), alive_vals, bottom=mort_vals, label='alive')

    #         for i, idx in enumerate(counts.index):
    #             total = counts.loc[idx].sum()

    #             txt = ""
    #             if total > 0:
    #                 pe = perc.loc[idx].get(1, 0)
    #                 pp = perc.loc[idx].get(0, 0)
    #                 txt = f"{pe:.1f}% 1\n{pp:.1f}% 0"

    #             plt.text(i, total, txt, ha='center', va='bottom', fontsize=9)

    #         plt.title(f"Distribution de MORTSTAT_2019 selon '{col}'\n"
    #                   f"p={p_value:.4g}, V={v:.3f}")
    #         plt.ylabel("Nombre")
    #         plt.xlabel(col)
    #         plt.xticks(rotation=45)
    #         plt.ylim(0, counts.sum(axis=1).max() * 1.15)
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig(f"crosstab/crosstab_{col}.png")
    #         plt.close()

    #         print(f"Graphique sauvegardé pour {col}")
    #     else:
    #         print(f"Graphique non sauvegardé pour {col} (pas assez significatif)")

    # print("\nVariables sauvegardées :")
    # for col, p_value, v in significant_vars:
    #     print(f"{col} -> p={p_value:.6f}, V={v:.4f}")