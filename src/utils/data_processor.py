import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List


class DataProcessor:
    """
    Utility class for data preprocessing operations.

    Attributes:
        scaler (StandardScaler): Scaler for feature normalization
        imputer (SimpleImputer): Imputer for handling missing values
        label_encoders (dict): Dictionary of label encoders for categorical variables
    """

    def __init__(self):
        """Initialize the DataProcessor with necessary preprocessing objects."""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}

    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Process the input data by handling missing values, encoding categorical variables,
        and scaling numerical features.

        Args:
            data (pd.DataFrame): Input data to be processed

        Returns:
            Tuple containing:
                - processed DataFrame
                - list of feature names
                - list of unique class labels
        """
        # Handle missing values in numerical features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])

        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if data.columns[-1] in categorical_cols:
            categorical_cols.remove(data.columns[-1])

        for column in categorical_cols:
            self.label_encoders[column] = LabelEncoder()
            data[column] = self.label_encoders[column].fit_transform(data[column])

        # Scale features
        feature_cols = data.columns[:-1]
        data[feature_cols] = self.scaler.fit_transform(data[feature_cols])

        return (
            data,
            feature_cols.tolist(),
            data.iloc[:, -1].unique().tolist()
        )