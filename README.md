# K-Means Clustering on Iris Dataset 🌸

An interactive machine learning project demonstrating K-Means clustering on the classic Iris flower dataset. The repository features a detailed Jupyter notebook tutorial/exercise and a premium Streamlit dashboard.

## Project Structure
* **`app.py`**: The Streamlit web application dashboard.
* **`kmeans_exercise.ipynb`**: Complete step-by-step Jupyter Notebook walking through data loading, feature selection, scaling, the Elbow Method, training, visualization, and validation.
* **`requirements.txt`**: Package dependencies for local setup and cloud deployment.
* **`income.csv`**: A supporting dataset used for tutorials on K-Means with scaling.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
If `streamlit` is not recognized globally on Windows, you can run it directly using the Python module syntax:
```bash
python -m streamlit run app.py
```

### 4. Open the Jupyter Notebook
To run and explore the notebook exercise locally:
```bash
jupyter notebook kmeans_exercise.ipynb
```

---

## Streamlit App Features
* **Interactive Parameters**: Dynamic adjustment of the cluster count ($k$) and feature scaling option.
* **Elbow Curve**: Instant visual feedback of WCSS (inertia) values.
* **Centroids Overlay**: Visualizes centroids mapped back to the original petal measurements.
* **Predictor**: Predicts the cluster for custom petal lengths and widths in real time.
* **Alignment Matrix**: Cross-tabulates predicted clusters against ground-truth classes.
