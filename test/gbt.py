import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class GBT():
    def __init__(self, num_estimators, max_depth=5, min_split=2, learning_rate=0.01, criterion='mse'):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_split = min_split
        self.num_estimators = num_estimators
        self.criterion = criterion
        self.models = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = np.zeros(len(y))
        if self.criterion != 'mae':
            self.criterion = 'mse'
        for _ in range(self.num_estimators):
            tree = RegressionTree(max_depth=self.max_depth, min_split=self.min_split, criterion=self.criterion)
            residual = y - y_pred
            tree.fit(X, residual)
            gamma = self.learning_rate * tree.predict(X)
            y_pred += gamma
            self.models.append(tree)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred

class RegressionTree():
    def __init__(self, max_depth=5, min_split=2, criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        def build_tree(X, y, depth):
            if depth >= self.max_depth or len(y) < self.min_samples_split:
                return {"leaf_value": np.mean(y)}
            best_split_point = self.find_best_split_point(X, y)
            if not best_split_point:
                return {"leaf_value": np.mean(y)}
            left_subtree = build_tree(best_split_point["left_X"], best_split_point["left_y"], depth + 1)
            right_subtree = build_tree(best_split_point["right_X"], best_split_point["right_y"], depth + 1)
            return {
                "feature_index": best_split_point["feature_index"],
                "split_value": best_split_point["split_value"],
                "left": left_subtree,
                "right": right_subtree
            }
        self.tree = build_tree(X, y, depth=0)

    def find_best_split_point(self, X, y):
        _, n_features = X.shape
        best_split = None
        best_error = float('inf')
        for index in range(n_features):
            values = X[:, index]
            dynamic_split = np.unique(values)
            for value in dynamic_split:
                left_indices = X[:, index] <= value
                right_indices = X[:, index] > value
                left_X, right_X = X[left_indices], X[right_indices]
                left_y, right_y = y[left_indices], y[right_indices]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                current_error = self.regErr(left_y, right_y, mode=self.criterion)
                if current_error < best_error:
                    best_error = current_error
                    best_split = {
                        "feature_index": index,
                        "split_value": value,
                        "left_X": left_X,
                        "left_y": left_y,
                        "right_X": right_X,
                        "right_y": right_y
                    }
        return best_split

    def predict(self, X):
        predictions = []
        for i in X:
            node = self.tree
            while "leaf_value" not in node:
                feature_index = node["feature_index"]
                split_value = node["split_value"]
                if i[feature_index] <= split_value:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["leaf_value"])
        return np.array(predictions)

    def regErr(self, left_y, right_y, mode='mse'):
        if mode == 'mae':
            left_mae = np.mean(np.abs(left_y - np.mean(left_y))) * len(left_y) if len(left_y) > 0 else 0
            right_mae = np.mean(np.abs(right_y - np.mean(right_y))) * len(right_y) if len(right_y) > 0 else 0
            return (left_mae + right_mae) / (len(left_y) + len(right_y))
        else:
            left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
            right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
            return (left_mse + right_mse) / (len(left_y) + len(right_y))


def load_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df['class'] = df['class'].astype('category').cat.codes
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

def train_and_save_model():
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GBT(num_estimators=20, max_depth=3, min_split=10, learning_rate=0.1, criterion='mse')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_class = np.round(y_pred).astype(int)
    y_pred_class = np.clip(y_pred_class, 0, 2)
    accuracy = np.mean(y_pred_class == y_test)
    print("Predicted values of y (rounded to nearest class):", y_pred_class)
    print("True values of y:", y_test)
    print("Classification Accuracy:", accuracy)
    with open('model/gbt_iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value (Class Labels)')
    plt.title('GBT Predictions vs True Values on Iris Dataset')
    plt.legend()
    plt.savefig('images/GBT_Predictions_Iris.png')
    plt.show()

def load_and_plot_model():
    if not os.path.exists('./model/gbt_iris_model.pkl'):
        print("No saved model found. Please train the model first.")
        return
    with open('./model/gbt_iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    y_pred = model.predict(X_test)
    y_pred_class = np.round(y_pred).astype(int)
    y_pred_class = np.clip(y_pred_class, 0, 2)
    accuracy = np.mean(y_pred_class == y_test)
    print("Classification Accuracy on the test dataset:", accuracy)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index (Test Set)')
    plt.ylabel('Target Value (Class Labels)')
    plt.title('GBT Predictions vs True Values on Test Set (Loaded Model)')
    plt.legend()
    plt.show()

def train_concrete_model(file_path='./test/Concrete_Data.xls'):
    if not os.path.exists(file_path):
        print(f"{file_path} file not found. Please ensure it is in the correct directory.")
        return
    dataset_name = os.path.basename(file_path).split('.')[0]
    print(f"Loading dataset: {file_path}")
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GBT(num_estimators=40, max_depth=5, min_split=10, learning_rate=0.08, criterion='mse')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"Evaluation Metrics for Dataset '{dataset_name}':")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    model_file_name = f'model/gbt_{dataset_name}_model.pkl'
    with open(model_file_name, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully as '{model_file_name}'")
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value (Compressive Strength)')
    plt.title(f'GBT Predictions vs True Values on Dataset: {dataset_name}')
    plt.legend()
    plot_file_name = f'images/GBT_Predictions_{dataset_name}.png'
    plt.savefig(plot_file_name)
    plt.show()
    print(f"Prediction plot saved as '{plot_file_name}'")

if __name__ == "__main__":
    while True:
        print("\nSelect an option:")
        print("1. Train Iris dataset, test it on test set and save model")
        print("2. Load saved Iris model and plot predictions")
        print("3. Train default Concrete_Data.xls dataset, test it, and save model")
        print("4. Train custom dataset (input file name), test it, and save model")
        print("q. Exit")

        choice = input("Enter your choice (1/2/3/4/q): ")

        if choice == '1':
            train_and_save_model()
        elif choice == '2':
            load_and_plot_model()
        elif choice == '3':
            train_concrete_model()  # default Concrete_Data.xls
        elif choice == '4':
            print("Please modify the GBT parameters in the code if needed before proceeding!")
            file_name = input("Enter the dataset file name (including extension, e.g., 'your_data.xls'): ")
            train_concrete_model(file_path=file_name)  # custom dataset
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or q.")
            print("Using default choice 1")
            train_and_save_model()