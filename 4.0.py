import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# ----------------- Utility Functions ----------------- #

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['target'] = df['exam_score'].apply(lambda x: 1 if x >= 50 else 0)
    df.drop(['student_id', 'exam_score'], axis=1, inplace=True)
    return df

def preprocess_data(df):
    y = df['target']
    X = df.drop('target', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_train_distributions(X_train, y_train, feature_names):
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train.values
    train_df[feature_names].hist(bins=20, figsize=(14, 10))
    plt.suptitle("Feature Distributions (Train Set)")
    plt.show()

def fuzzy_triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def add_fuzzy_features(X_scaled, feature_names):
    df_fuzzy = pd.DataFrame(X_scaled, columns=feature_names)
    for feature in feature_names[:2]:  # Use first 2 numeric features
        val = df_fuzzy[feature]
        df_fuzzy[f'{feature}_low'] = fuzzy_triangular(val, 0, 0, 0.5)
        df_fuzzy[f'{feature}_med'] = fuzzy_triangular(val, 0.25, 0.5, 0.75)
        df_fuzzy[f'{feature}_high'] = fuzzy_triangular(val, 0.5, 1, 1)
    return df_fuzzy

# ----------------- Model Training Functions ----------------- #

def hill_climbing_decision_tree(X_train, y_train):
    best_score = 0
    best_depth = None
    current_depth = 1
    search_path = []

    while True:
        clf = DecisionTreeClassifier(max_depth=current_depth, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        search_path.append((current_depth, score))

        if score > best_score:
            best_score = score
            best_depth = current_depth
            current_depth += 1
        else:
            break

    return best_depth, best_score, search_path

def grid_search_decision_tree(X_train, y_train, max_depth_range=10):
    results = []
    for d in range(1, max_depth_range + 1):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_train, y_train)
        results.append((d, acc))
    return results

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\nEvaluation on Test Set:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:\n", cm)
    return y_pred

def visualize_tree(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Fail", "Pass"], filled=True)
    plt.title("Final Tuned Decision Tree")
    plt.show()

# ----------------- Main Runner ----------------- #

def main():
    # Load and preprocess
    data = load_and_clean_data("./student_habits_performance.csv")
    X, y = preprocess_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

    # EDA on train
    plot_train_distributions(X_train_scaled, y_train, X.columns)

    # Fuzzy features
    fuzzy_train = add_fuzzy_features(X_train_scaled, X.columns)
    fuzzy_test = add_fuzzy_features(X_test_scaled, X.columns)

    # --- Hill Climbing ---
    best_depth_hc, best_score_hc, hc_path = hill_climbing_decision_tree(fuzzy_train, y_train)
    hc_depths, hc_scores = zip(*hc_path)

    print(f"\nHill-Climbing Best Depth: {best_depth_hc}, Accuracy: {best_score_hc:.4f}")
    print(f"Hill-Climbing Evaluations: {len(hc_path)}")

    # --- Grid Search ---
    grid_results = grid_search_decision_tree(fuzzy_train, y_train)
    gs_depths, gs_scores = zip(*grid_results)

    best_depth_gs = max(grid_results, key=lambda x: x[1])[0]
    best_score_gs = max(grid_results, key=lambda x: x[1])[1]

    print(f"\nGrid Search Best Depth: {best_depth_gs}, Accuracy: {best_score_gs:.4f}")
    print(f"Grid Search Evaluations: {len(grid_results)}")

    # --- Visual Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(gs_depths, gs_scores, marker='o', label='Grid Search')
    plt.plot(hc_depths, hc_scores, marker='x', linestyle='--', label='Hill-Climbing')
    plt.xlabel("Tree Depth")
    plt.ylabel("Train Accuracy")
    plt.title("Hill-Climbing vs. Grid Search")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Final Model ---
    final_clf = DecisionTreeClassifier(max_depth=best_depth_hc, random_state=42)
    final_clf.fit(fuzzy_train, y_train)

    # --- Evaluation ---
    y_pred = evaluate_model(final_clf, fuzzy_test, y_test)

    # --- Save + Visualize Tree ---
    joblib.dump(final_clf, "decision_tree_model.pkl")
    visualize_tree(final_clf, fuzzy_train.columns)
    # --- Load and Predict ---
    loaded_model = joblib.load("decision_tree_model.pkl")

#   --> GUI FROM HERE <--
# ----------------- GUI Code ----------------- #
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

final_clf = None
feature_names = []
X_train_scaled = None
X_test_scaled = None
y_train = None
y_test = None
fuzzy_train = None
fuzzy_test = None


def fuzzy_triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))


def add_fuzzy_features(X_scaled, base_features):
    df_fuzzy = pd.DataFrame(X_scaled, columns=base_features)
    for feature in base_features[:2]:
        val = df_fuzzy[feature]
        df_fuzzy[f'{feature}_low'] = fuzzy_triangular(val, 0, 0, 0.5)
        df_fuzzy[f'{feature}_med'] = fuzzy_triangular(val, 0.25, 0.5, 0.75)
        df_fuzzy[f'{feature}_high'] = fuzzy_triangular(val, 0.5, 1, 1)
    return df_fuzzy


def hill_climbing_decision_tree(X_train, y_train):
    best_score = 0
    best_depth = None
    current_depth = 1
    search_path = []

    while True:
        clf = DecisionTreeClassifier(max_depth=current_depth, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        search_path.append((current_depth, score))

        if score > best_score:
            best_score = score
            best_depth = current_depth
            current_depth += 1
        else:
            break

    return best_depth, best_score, search_path


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, cm


def process_dataset(path):
    global final_clf, feature_names, X_train_scaled, X_test_scaled, y_train, y_test, fuzzy_train, fuzzy_test
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    if 'exam_score' not in df.columns or 'student_id' not in df.columns:
        messagebox.showerror("Error", "Dataset must contain 'student_id' and 'exam_score' columns.")
        return
    df['target'] = df['exam_score'].apply(lambda x: 1 if x >= 50 else 0)
    df.drop(['student_id', 'exam_score'], axis=1, inplace=True)

    y = df['target']
    X = df.drop('target', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    if len(feature_names) < 10:
        messagebox.showerror("Error", "Dataset must have at least 10 features after encoding.")
        return

    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train = y_train_temp
    y_test = y_test_temp

    fuzzy_train = add_fuzzy_features(X_train_scaled, feature_names)
    fuzzy_test = add_fuzzy_features(X_test_scaled, feature_names)

    best_depth, _, _ = hill_climbing_decision_tree(fuzzy_train, y_train)
    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    clf.fit(fuzzy_train, y_train)
    final_clf = clf
    messagebox.showinfo("Success", f"Model trained with max_depth={best_depth} and ready for visualization.")
    btn_tree.pack(pady=10)
    btn_features.pack(pady=10)
    btn_evaluate.pack(pady=10)


def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        process_dataset(file_path)


def display_plot(fig):
    window = tk.Toplevel(app)
    window.title("Visualization")
    canvas = FigureCanvasTkAgg(fig, master=window)
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


def show_tree():
    if final_clf is None:
        messagebox.showwarning("Error", "Train a model first by loading a dataset.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    try:
        plot_tree(final_clf, feature_names=final_clf.feature_names_in_, class_names=["Fail", "Pass"], filled=True, ax=ax)
    except AttributeError:
        plot_tree(final_clf, filled=True, ax=ax)
    display_plot(fig)


def show_feature_distribution():
    if X_train_scaled is None or y_train is None:
        messagebox.showwarning("Error", "Load and process a dataset first.")
        return

    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['target'] = y_train.values
    fig, axs = plt.subplots(len(feature_names) // 3 + 1, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i, col in enumerate(feature_names):
        if i < len(axs):
            axs[i].hist(train_df[col], bins=20, color='skyblue', edgecolor='black')
            axs[i].set_title(col)

    plt.tight_layout()
    display_plot(fig)


def show_evaluation():
    if final_clf is None or fuzzy_test is None or y_test is None:
        messagebox.showwarning("Error", "You need to train the model first.")
        return

    acc, prec, rec, cm = evaluate_model(final_clf, fuzzy_test, y_test)
    result = f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nConfusion Matrix:\n{cm}"
    messagebox.showinfo("Model Evaluation", result)


app = tk.Tk()
app.title("AI Project Visualizer")
app.geometry("1200x900")

btn_load = tk.Button(app, text="Load CSV Dataset", command=load_csv, width=25, height=2)
btn_load.pack(pady=20)

btn_tree = tk.Button(app, text="Visualize Decision Tree", command=show_tree, width=25, height=2)
btn_features = tk.Button(app, text="Show Feature Distributions", command=show_feature_distribution, width=25, height=2)
btn_evaluate = tk.Button(app, text="Evaluate Model on Test Set", command=show_evaluation, width=25, height=2)

app.mainloop()
        
if __name__ == "__main__":
    main()



import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ----------------- Load Model & Scaler ----------------- #

try:
    model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    print("ERROR: Required files not found.")
    raise e

# ----------------- Dummy Training Data (for visualizations) ----------------- #

try:
    train_data = pd.read_csv("student_habits_performance.csv").dropna().drop_duplicates()
    train_data['target'] = train_data['exam_score'].apply(lambda x: 1 if x >= 50 else 0)
    train_data.drop(['student_id', 'exam_score'], axis=1, inplace=True)
    X = pd.get_dummies(train_data.drop('target', axis=1), drop_first=True)
    y = train_data['target']
    X_scaled = scaler.transform(X)
    fuzzy_train = pd.DataFrame(X_scaled, columns=X.columns)
    for feature in X.columns[:2]:
        fuzzy_train[f'{feature}_low'] = np.maximum(0, np.minimum((fuzzy_train[feature] - 0) / (0.5 - 0), (0.5 - fuzzy_train[feature]) / (0.5 - 0)))
        fuzzy_train[f'{feature}_med'] = np.maximum(0, np.minimum((fuzzy_train[feature] - 0.25) / (0.5 - 0.25), (0.75 - fuzzy_train[feature]) / (0.75 - 0.5)))
        fuzzy_train[f'{feature}_high'] = np.maximum(0, np.minimum((fuzzy_train[feature] - 0.5) / (1 - 0.5), (1 - fuzzy_train[feature]) / (1 - 0.5)))
except:
    fuzzy_train = None
    X = None
    y = None

# ----------------- GUI Setup ----------------- #

root = tk.Tk()
root.title("📚 Student Performance Predictor")

feature_names = [
    'study_hours',
    'sleep_hours',
    'attendance_rate',
    'has_tutor_Yes',
    'school_support_Yes',
    'internet_Yes'
]

entries = {}

def fuzzy_triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def add_fuzzy_features_single(sample, feature_names):
    df = pd.DataFrame([sample], columns=feature_names)
    for feature in feature_names[:2]:
        val = df[feature]
        df[f'{feature}_low'] = fuzzy_triangular(val, 0, 0, 0.5)
        df[f'{feature}_med'] = fuzzy_triangular(val, 0.25, 0.5, 0.75)
        df[f'{feature}_high'] = fuzzy_triangular(val, 0.5, 1, 1)
    return df

def predict():
    try:
        values = [float(entries[f].get()) for f in feature_names]
        scaled = scaler.transform([values])
        fuzzy_input = add_fuzzy_features_single(scaled[0], feature_names)
        result = model.predict(fuzzy_input)[0]
        text = "✅ PASS" if result == 1 else "❌ FAIL"
        messagebox.showinfo("Prediction", f"The student will likely {text}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def show_tree():
    if fuzzy_train is None:
        messagebox.showerror("Error", "Training data not available.")
        return
    plt.figure(figsize=(18, 10))
    plot_tree(model, feature_names=fuzzy_train.columns, class_names=["Fail", "Pass"], filled=True)
    plt.title("Decision Tree")
    plt.show()

def show_distributions():
    if X is None:
        messagebox.showerror("Error", "Training data not available.")
        return
    df = pd.DataFrame(scaler.transform(X), columns=X.columns)
    df.hist(bins=20, figsize=(14, 10))
    plt.suptitle("Feature Distributions (Train Set)")
    plt.show()

def show_comparison():
    depths = list(range(1, 11))
    hc_scores = []
    gs_scores = []

    # Hill-climbing simulation
    best = 0
    for d in depths:
        clf = joblib.load("decision_tree_model.pkl")
        clf.set_params(max_depth=d)
        clf.fit(fuzzy_train, y)
        acc = clf.score(fuzzy_train, y)
        gs_scores.append(acc)
        if acc > best:
            best = acc
            hc_scores.append(acc)
        else:
            break

    plt.figure(figsize=(10, 6))
    plt.plot(depths[:len(gs_scores)], gs_scores, marker='o', label='Grid Search')
    plt.plot(depths[:len(hc_scores)], hc_scores, marker='x', linestyle='--', label='Hill Climbing')
    plt.xlabel("Tree Depth")
    plt.ylabel("Train Accuracy")
    plt.title("Hill-Climbing vs. Grid Search")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------- GUI Layout ----------------- #

for i, fname in enumerate(feature_names):
    tk.Label(root, text=fname).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[fname] = entry

tk.Button(root, text="🔮 Predict", command=predict, bg="lightgreen").grid(row=len(feature_names), columnspan=2, pady=10)
tk.Button(root, text="🌳 Show Tree", command=show_tree).grid(row=len(feature_names)+1, column=0, pady=5)
tk.Button(root, text="📊 Distributions", command=show_distributions).grid(row=len(feature_names)+1, column=1, pady=5)
tk.Button(root, text="📈 Compare Search", command=show_comparison).grid(row=len(feature_names)+2, columnspan=2, pady=10)

root.mainloop()
