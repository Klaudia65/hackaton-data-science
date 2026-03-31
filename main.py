import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
    


if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('data/data.csv')

    print(data.head())