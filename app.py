import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration for a wider, modern layout
st.set_page_config(
    page_title="Iris K-Means Clustering Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium styling, modern fonts, and card containers
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #f9f9fb;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #eef0f5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-header {
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allowed_html=True)

# ----------------- Data Loading -----------------
@st.cache_data
def get_iris_data():
    raw_iris = load_iris()
    df_raw = pd.DataFrame(raw_iris.data, columns=raw_iris.feature_names)
    df_raw['species'] = [raw_iris.target_names[t] for t in raw_iris.target]
    df_raw['target'] = raw_iris.target
    return raw_iris, df_raw

iris_raw, df_iris = get_iris_data()
df_features = df_iris[['petal length (cm)', 'petal width (cm)']].copy()

# ----------------- Sidebar Controls -----------------
st.sidebar.markdown('<div class="sidebar-header">Configuration ⚙️</div>', unsafe_allowed_html=True)
st.sidebar.markdown("Configure the K-Means parameters below:")

# Slider for number of clusters (K)
k_val = st.sidebar.slider(
    "Number of Clusters (k)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Select the value of k for training the K-Means algorithm."
)

# Toggle to enable/disable feature scaling
scaling_enabled = st.sidebar.checkbox(
    "Enable Feature Scaling",
    value=True,
    help="Scale the features to mean=0 and std=1 before clustering."
)

# Quick info in the sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App:**\n\n"
    "This app demonstrates K-Means clustering using "
    "petal features of the famous Iris dataset. "
    "Unsupervised models ignore labels and group samples based on similarity."
)

# ----------------- Page Header -----------------
st.markdown('<div class="main-title">Iris Petal Clustering Dashboard</div>', unsafe_allowed_html=True)
st.markdown('<div class="subtitle">An interactive visual guide to unsupervised machine learning using K-Means</div>', unsafe_allowed_html=True)

# ----------------- Preprocessing & Fitting -----------------
# Scaler initialization
scaler = StandardScaler()
if scaling_enabled:
    scaled_data = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_data, columns=df_features.columns)
    training_data = df_scaled
else:
    training_data = df_features

# Fit the K-Means model for the chosen k
kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
cluster_labels = kmeans_model.fit_predict(training_data)
df_iris['predicted_cluster'] = cluster_labels

# ----------------- Layout Tabs -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration", 
    "📐 Elbow Method", 
    "🎨 K-Means Clustering", 
    "🔮 Cluster Predictor",
    "📋 Evaluation Matrix"
])

# ----------------- Tab 1: Data Exploration -----------------
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("Explore the Iris Dataset")
        st.write(
            "The Iris flower dataset contains 150 samples from three species of Iris "
            "(Iris setosa, Iris virginica, and Iris versicolor). Four features were "
            "measured from each sample: the length and the width of the sepals and petals."
        )
        st.write(
            "For this exercise, we focus strictly on the **petal length** and **petal width** features to simplify "
            "cluster visualization."
        )
        st.markdown('</div>', unsafe_allowed_html=True)
        
        st.subheader("Raw Dataset View")
        st.dataframe(df_iris, use_container_width=True, height=250)
        
    with col2:
        st.subheader("Petal Features Distribution (Raw)")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df_iris['petal length (cm)'],
            df_iris['petal width (cm)'],
            color='#8c96c6',
            edgecolor='k',
            alpha=0.8,
            s=60
        )
        ax.set_title("Petal Length vs. Petal Width")
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

# ----------------- Tab 2: Elbow Method -----------------
with tab2:
    col_el1, col_el2 = st.columns([1, 1.2])
    
    with col_el1:
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("What is the Elbow Method?")
        st.write(
            "The Elbow Method is a heuristic used to find the optimal number of clusters in a dataset. "
            "It plots the Within-Cluster Sum of Squares (WCSS / Inertia) against the number of clusters $k$."
        )
        st.write(
            "The **WCSS** measures the sum of squared distances between each point and its assigned cluster centroid. "
            "As $k$ increases, WCSS naturally decreases (reaching 0 when $k$ equals the number of samples). "
            "We look for an **elbow point**—a point where the rate of WCSS decrease slows down dramatically."
        )
        st.markdown('</div>', unsafe_allowed_html=True)
        
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("Feature Scaling Impact")
        st.write(
            "Feature scaling ensures variables with different magnitudes contribute equally to distances. "
            "Since petal length and width are both in centimeters with similar ranges, scaling is not strictly "
            "necessary here, but it is highly recommended as a standard preprocessing step."
        )
        st.write(f"**Scaling Enabled:** `{scaling_enabled}`")
        st.markdown('</div>', unsafe_allowed_html=True)
        
    with col_el2:
        # Calculate WCSS/Inertia for k in 1..10
        wcss = []
        k_rng = range(1, 11)
        for k in k_rng:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km_temp.fit(training_data)
            wcss.append(km_temp.inertia_)
            
        st.subheader("Elbow Curve (WCSS vs. k)")
        fig_elb, ax_elb = plt.subplots(figsize=(8, 5))
        ax_elb.plot(k_rng, wcss, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label="WCSS")
        
        # Highlight current K
        ax_elb.scatter(k_val, wcss[k_val - 1], color='red', s=150, zorder=5, label=f"Selected k={k_val}")
        
        ax_elb.set_title("WCSS / Inertia across k Values")
        ax_elb.set_xlabel("Number of Clusters (k)")
        ax_elb.set_ylabel("Inertia (WCSS)")
        ax_elb.set_xticks(k_rng)
        ax_elb.grid(True, linestyle='--', alpha=0.5)
        ax_elb.legend()
        st.pyplot(fig_elb)

# ----------------- Tab 3: K-Means Clustering -----------------
with tab3:
    col_cl1, col_cl2 = st.columns([1.5, 1])
    
    with col_cl1:
        st.subheader(f"K-Means Clustering Visual (k = {k_val})")
        
        # Plot clusters and centroids
        fig_cl, ax_cl = plt.subplots(figsize=(9, 7))
        
        # Generate custom colors palette
        cmap = plt.cm.get_cmap('tab10', k_val)
        
        for cluster_id in range(k_val):
            subset = df_iris[df_iris['predicted_cluster'] == cluster_id]
            ax_cl.scatter(
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                color=cmap(cluster_id),
                label=f'Cluster {cluster_id}',
                edgecolor='k',
                alpha=0.8,
                s=60
            )
            
        # Centroids logic
        centroids = kmeans_model.cluster_centers_
        if scaling_enabled:
            # Map back to original scale for plotting
            original_centroids = scaler.inverse_transform(centroids)
        else:
            original_centroids = centroids
            
        ax_cl.scatter(
            original_centroids[:, 0],
            original_centroids[:, 1],
            color='red',
            marker='X',
            s=250,
            edgecolor='k',
            linewidths=1.5,
            label='Centroids',
            zorder=10
        )
        
        ax_cl.set_title("Clustered Petal Features with Centroids")
        ax_cl.set_xlabel("Petal Length (cm)")
        ax_cl.set_ylabel("Petal Width (cm)")
        ax_cl.legend()
        ax_cl.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_cl)
        
    with col_cl2:
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("Model Summary")
        st.metric("Total Iterations", kmeans_model.n_iter_)
        st.metric("Final Inertia (WCSS)", f"{kmeans_model.inertia_:.4f}")
        st.markdown('</div>', unsafe_allowed_html=True)
        
        st.subheader("Cluster Centroid Coordinates")
        centroids_df = pd.DataFrame(
            original_centroids,
            columns=['Petal Length (cm)', 'Petal Width (cm)'],
            index=[f"Cluster {i}" for i in range(k_val)]
        )
        st.dataframe(centroids_df, use_container_width=True)

# ----------------- Tab 4: Cluster Predictor -----------------
with tab4:
    st.subheader("Interactive Flower Cluster Predictor 🔮")
    st.write(
        "Move the sliders to define petal length and width features. "
        "The model will scale the features (if enabled) and predict which cluster "
        "the new sample falls into."
    )
    
    col_pr1, col_pr2 = st.columns([1, 1.2])
    
    with col_pr1:
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("Input Petal Features")
        
        input_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5, 0.1)
        input_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)
        
        st.markdown('</div>', unsafe_allowed_html=True)
        
        # Perform prediction
        new_sample = np.array([[input_length, input_width]])
        if scaling_enabled:
            scaled_sample = scaler.transform(new_sample)
            predicted_c = kmeans_model.predict(scaled_sample)[0]
        else:
            predicted_c = kmeans_model.predict(new_sample)[0]
            
        st.success(f"### Predicted Group: **Cluster {predicted_c}**")
        
    with col_pr2:
        st.subheader("New Sample Placement on Plot")
        
        # Plot existing clusters and highlight new sample
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        cmap_pr = plt.cm.get_cmap('tab10', k_val)
        
        for cluster_id in range(k_val):
            subset = df_iris[df_iris['predicted_cluster'] == cluster_id]
            ax_pr.scatter(
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                color=cmap_pr(cluster_id),
                alpha=0.4,
                s=40
            )
            
        # Draw the new point as a big yellow star
        ax_pr.scatter(
            input_length,
            input_width,
            color='yellow',
            edgecolor='black',
            marker='*',
            s=350,
            label='New Sample Input',
            zorder=12
        )
        
        # Centroids
        ax_pr.scatter(
            original_centroids[:, 0],
            original_centroids[:, 1],
            color='red',
            marker='X',
            s=120,
            edgecolor='k',
            label='Centroids'
        )
        
        ax_pr.set_xlabel("Petal Length (cm)")
        ax_pr.set_ylabel("Petal Width (cm)")
        ax_pr.legend()
        ax_pr.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_pr)

# ----------------- Tab 5: Evaluation Matrix -----------------
with tab5:
    st.subheader("Evaluation Matrix vs. Actual Labels")
    st.write(
        "Here, we compare the unsupervised clusters with the actual species targets. "
        "Note that unsupervised learning algorithms train without target labels. "
        "This evaluation is strictly to analyze cluster alignment with the natural classes."
    )
    
    col_ev1, col_ev2 = st.columns([1.2, 1])
    
    with col_ev1:
        st.subheader("Cross-Tabulation Matrix")
        crosstab_df = pd.crosstab(df_iris['species'], df_iris['predicted_cluster'])
        st.dataframe(crosstab_df, use_container_width=True)
        
    with col_ev2:
        st.markdown('<div class="card">', unsafe_allowed_html=True)
        st.subheader("Evaluation Summary")
        
        # Analyze and provide qualitative feedback based on chosen k
        if k_val == 3:
            st.info(
                "💡 **Analysis for k = 3:**\n\n"
                "- **Setosa** is perfectly isolated into a single cluster.\n"
                "- **Versicolor** and **Virginica** have slight overlaps due to proximity "
                "in their petal sizes, which closely mirrors the botanical overlapping characteristics."
            )
        else:
            st.warning(
                f"💡 **Analysis for k = {k_val}:**\n\n"
                "The actual dataset consists of exactly **3 natural species** (Setosa, Versicolor, Virginica). "
                f"Using $k={k_val}$ clusters over-segments or under-segments the dataset, resulting in "
                "poor alignment with biological groups. Check out the Elbow Plot in Tab 2 to see why $k=3$ is optimal!"
            )
        st.markdown('</div>', unsafe_allowed_html=True)
