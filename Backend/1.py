
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
url_class_df = pd.read_csv(
    r'C:\Users\abinj\ABXx\VS Code\AI_ Based_Multimodel_Threat_Detection\Backend\Dataset\URL Classification.csv',
    header=None,  # No header row in file
    names=['id', 'url', 'type']  # Specify column names manually
)

