from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
import pandas as pd
import joblib

# Load iris dataset (example dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Let's pretend the first column is categorical for demonstration (usually this is more relevant to non-numeric data)
# We will convert the first feature column into categorical data just for this example
X = pd.DataFrame(X)
X[0] = X[0].astype("category")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Define preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            [0],
        ),  # Apply OneHotEncoder to the first column (categorical)
        (
            "num",
            StandardScaler(),
            [1, 2, 3],
        ),  # Apply StandardScaler to the remaining columns (numerical)
    ]
)

# Create the pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),  # First step: preprocessing
        ("pca", PCA(n_components=2)),  # Second step: PCA
        (
            "decision_tree",
            DecisionTreeClassifier(),
        ),  # Third step: DecisionTreeClassifier
    ]
)

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Make predictions and evaluate
from sklearn.metrics import accuracy_score

y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save the pipeline to a file
joblib.dump(pipe, "../app/model_pipeline.joblib")
