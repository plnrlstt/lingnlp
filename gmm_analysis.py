import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def find_optimal_clusters(data, file_name):
    """
    Fits GMM for a range of clusters and plots AIC/BIC to find the optimal number.
    """
    # Reshape data for GMM (needs to be a 2D array)
    X = data.values.reshape(-1, 1)

    n_components = np.arange(1, 11)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
              for n in n_components]

    plt.figure(figsize=(8, 6))
    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.title(f'AIC and BIC for {file_name}')
    plt.xlabel('Number of components')
    plt.ylabel('Score')
    plt.legend(loc='best')
    
    # Save the plot
    plot_filename = f"aic_bic_{file_name.replace('.csv', '.png')}"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

    bic_scores = [m.bic(X) for m in models]
    aic_scores = [m.aic(X) for m in models]

    optimal_n_bic = n_components[np.argmin(bic_scores)]
    optimal_n_aic = n_components[np.argmin(aic_scores)]

    print(f"For {file_name}:")
    print(f"Optimal number of clusters (BIC): {optimal_n_bic}")
    print(f"Optimal number of clusters (AIC): {optimal_n_aic}")
    print("-" * 30)

# Load the datasets
try:
    df_original = pd.read_csv("c:\\Users\\kpsst\\Desktop\\Uni\\Linguistics in Modern NLP\\LingNLP\\taxonomy_original_fullnames.csv", header=0)
    find_optimal_clusters(df_original['Cluster'], "taxonomy_original_fullnames.csv")
except FileNotFoundError:
    print("Error: 'taxonomy_original_fullnames.csv' not found.")

try:
    df_all7500 = pd.read_csv("c:\\Users\\kpsst\\Desktop\\Uni\\Linguistics in Modern NLP\\LingNLP\\taxonomy_all7500_fullnames.csv", header=None, names=['language', 'cluster'])
    find_optimal_clusters(df_all7500['cluster'], "taxonomy_all7500_fullnames.csv")
except FileNotFoundError:
    print("Error: 'taxonomy_all7500_fullnames.csv' not found.")
