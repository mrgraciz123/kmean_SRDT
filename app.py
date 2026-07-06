import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration for a wider, modern layout
st.set_page_config(
    page_title="Iris K-Means Clustering Space",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Visual Styles (Custom Dark / Glassmorphic CSS) -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Override font family and global backgrounds */
    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', -apple-system, sans-serif;
        background: radial-gradient(circle at 30% 20%, #151624 0%, #07080f 100%) !important;
        color: #e0e4ec !important;
    }
    
    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #0b0c14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Top banner and titles */
    .cyber-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff007f 0%, #7000ff 50%, #00f0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0px 0px 30px rgba(112, 0, 255, 0.35));
    }
    
    .cyber-subtitle {
        font-size: 1.25rem;
        color: #8c9bb4;
        font-weight: 300;
        margin-bottom: 2.2rem;
        letter-spacing: 0.5px;
    }
    
    /* Neon glowing card style */
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 18px !important;
        padding: 1.75rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6) !important;
        margin-bottom: 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .glass-card:hover {
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 240, 255, 0.15) !important;
        transform: translateY(-2px);
    }
    
    /* Custom tab headers styling override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #8c9bb4;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(112, 0, 255, 0.15) !important;
        border: 1px solid #7000ff !important;
        color: #00f0ff !important;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-weight: 700;
        font-size: 1.35rem;
        background: linear-gradient(135deg, #00f0ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Prediction output card */
    .prediction-box {
        text-align: center;
        padding: 2.25rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255, 0, 127, 0.12) 0%, rgba(112, 0, 255, 0.12) 100%);
        border: 2px dashed #ff007f;
        box-shadow: 0 0 35px rgba(255, 0, 127, 0.25);
        color: #ffffff;
        margin-top: 1.5rem;
        transition: all 0.4s ease;
    }
    .prediction-box h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ff007f;
        text-shadow: 0 0 15px rgba(255, 0, 127, 0.5);
    }
    
    /* Floating animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 240, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Matplotlib Dark Theme Helper -----------------
def apply_dark_theme(fig, ax, title_text="", xlabel="", ylabel=""):
    fig.patch.set_facecolor('#07080f')
    ax.set_facecolor('#07080f')
    
    # Border spines customization
    for spine in ax.spines.values():
        spine.set_color('#1e2030')
        spine.set_linewidth(1.2)
        
    ax.tick_params(colors='#8f9cae', which='both', labelsize=10)
    ax.xaxis.label.set_color('#8f9cae')
    ax.yaxis.label.set_color('#8f9cae')
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    
    ax.set_title(title_text, color='#00f0ff', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color='#171926', linestyle=':', alpha=0.7)

# Neon colors list for clustering
NEON_COLORS = [
    '#00f0ff',  # Cyber Cyan
    '#ff007f',  # Neon Orchid Pink
    '#7000ff',  # Electric Violet
    '#ffaa00',  # Electric Gold
    '#00ff66',  # Neon Mint Green
    '#ff3300',  # Neon Flame Red
    '#ffff00',  # Radioactive Yellow
    '#ff00ff',  # Acid Fuchsia
    '#00ffff',  # Aqua Marine
    '#ffffff'   # Vaporwave White
]

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
st.sidebar.markdown('<div class="sidebar-header">Configurations ⚡</div>', unsafe_allow_html=True)
st.sidebar.markdown("Fine-tune model attributes and watch the visualizations react in real-time:")

# Slider for k clusters
k_val = st.sidebar.slider(
    "Clusters count (k)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Select the value of k for partition clustering."
)

# Scaling Toggle
scaling_enabled = st.sidebar.checkbox(
    "Standardize Features (StandardScaler)",
    value=True,
    help="Highly recommended for distance-sensitive algorithms like K-Means."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Interactive Engine")
st.sidebar.caption(
    "Standardizing transforms features to have zero mean and unit variance. "
    "K-Means computes Euclidean distance, making scaling essential when features use different metrics."
)

# ----------------- Top Header -----------------
st.markdown('<div class="cyber-title">Iris K-Means Clustering Space</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subtitle">An immersive, interactive visual sandbox for unsupervised machine learning</div>', unsafe_allow_html=True)

# ----------------- Preprocessing & K-Means Fitting -----------------
scaler = StandardScaler()
if scaling_enabled:
    scaled_data = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_data, columns=df_features.columns)
    training_data = df_scaled
else:
    training_data = df_features

kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
cluster_labels = kmeans_model.fit_predict(training_data)
df_iris['predicted_cluster'] = cluster_labels

# ----------------- Dashboard Layout (Tabs) -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Exploration",
    "📐 The Elbow Search",
    "🎨 Neon Clusters",
    "🔮 Spatial Predictor",
    "📋 Evaluation Bench"
])

# ----------------- Tab 1: Dataset Exploration -----------------
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Unsupervised Data Exploration")
        st.write(
            "In machine learning, exploratory data analysis is crucial. Here we load the famous "
            "**Iris Flower Dataset**. While it contains four measurements, we drop the sepal length "
            "and width features. Clustering petal length and width exposes a strong botanical structure "
            "that the algorithm will attempt to segregate completely blindly."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Raw Feature Table")
        st.dataframe(df_iris, use_container_width=True, height=270)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Raw Distribution View")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        apply_dark_theme(fig, ax, "Petal Dimensions Scatter", "Petal Length (cm)", "Petal Width (cm)")
        
        # Plot original unclustered data with soft cyber glowing color
        ax.scatter(
            df_iris['petal length (cm)'],
            df_iris['petal width (cm)'],
            color='#8b9bb4',
            edgecolor='#e0e4ec',
            alpha=0.6,
            s=80,
            linewidth=0.8,
            zorder=3
        )
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Tab 2: The Elbow Search -----------------
with tab2:
    col_el1, col_el2 = st.columns([1, 1.3])
    
    with col_el1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("The WCSS & Elbow Logic")
        st.write(
            "K-Means works by minimizing distances. The metric minimized is **Within-Cluster Sum of Squares (WCSS)**, "
            "also known as **Inertia**. A lower inertia implies tight, cohesive clusters."
        )
        st.write(
            "By plotting WCSS against the number of clusters $k$, we observe an **Elbow**. "
            "The optimal number of clusters resides at this bending point. Beyond it, adding more clusters yields "
            "negligible improvements in error reduction."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Scaling Decision Details")
        st.write(
            "Because K-Means calculates straight-line distances, scaling is highly recommended. "
            "If your features have significantly different scales, the feature with the largest scale "
            "dominates the cluster configuration. "
        )
        st.caption(f"**Current preprocessing standardizing state:** `{scaling_enabled}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_el2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Calculate WCSS/Inertia for k in 1..10
        wcss = []
        k_rng = range(1, 11)
        for k in k_rng:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km_temp.fit(training_data)
            wcss.append(km_temp.inertia_)
            
        fig_elb, ax_elb = plt.subplots(figsize=(8, 5))
        apply_dark_theme(fig_elb, ax_elb, "Inertia (WCSS) vs Number of Clusters", "Clusters (k)", "Inertia (WCSS)")
        
        # Plot elbow line
        ax_elb.plot(k_rng, wcss, marker='o', linestyle='-', color='#00f0ff', linewidth=2.5, markersize=7, zorder=3)
        
        # Highlight selected k value
        ax_elb.scatter(
            k_val,
            wcss[k_val - 1],
            color='#ff007f',
            s=220,
            edgecolor='#ffffff',
            linewidths=2.0,
            zorder=6,
            label=f"Selected k = {k_val}"
        )
        ax_elb.legend(facecolor='#07080f', edgecolor='#1e2030', labelcolor='#8f9cae')
        ax_elb.set_xticks(k_rng)
        st.pyplot(fig_elb)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Tab 3: Neon Clusters -----------------
with tab3:
    col_cl1, col_cl2 = st.columns([1.6, 1])
    
    with col_cl1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Plot clusters and centroids
        fig_cl, ax_cl = plt.subplots(figsize=(9, 7))
        apply_dark_theme(
            fig_cl, 
            ax_cl, 
            f"Partition Results (k = {k_val} clusters)", 
            "Petal Length (cm)", 
            "Petal Width (cm)"
        )
        
        # Draw clusters
        for cluster_id in range(k_val):
            subset = df_iris[df_iris['predicted_cluster'] == cluster_id]
            ax_cl.scatter(
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                color=NEON_COLORS[cluster_id % len(NEON_COLORS)],
                label=f'Cluster {cluster_id}',
                edgecolor='#07080f',
                alpha=0.85,
                s=80,
                linewidth=0.8,
                zorder=4
            )
            
        # Centroids logic
        centroids = kmeans_model.cluster_centers_
        if scaling_enabled:
            # Map back to original scale for plotting
            original_centroids = scaler.inverse_transform(centroids)
        else:
            original_centroids = centroids
            
        # Draw centroids as glowing neon target crosshairs
        ax_cl.scatter(
            original_centroids[:, 0],
            original_centroids[:, 1],
            color='#ffffff',
            marker='X',
            s=280,
            edgecolor='#ff007f',
            linewidths=2.5,
            label='Centroids',
            zorder=10
        )
        
        ax_cl.legend(facecolor='#07080f', edgecolor='#1e2030', labelcolor='#8f9cae')
        st.pyplot(fig_cl)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_cl2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model Status")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Iterations", kmeans_model.n_iter_)
        with col_m2:
            st.metric("Total WCSS", f"{kmeans_model.inertia_:.3f}")
            
        st.write("---")
        st.markdown("### Cluster Centroids Table")
        st.caption("Coordinates represent the geometric centers of the clusters in terms of raw centimeters.")
        
        centroids_df = pd.DataFrame(
            original_centroids,
            columns=['Petal Length (cm)', 'Petal Width (cm)'],
            index=[f"Cluster {i}" for i in range(k_val)]
        )
        st.dataframe(centroids_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Tab 4: Spatial Predictor -----------------
with tab4:
    st.subheader("Real-Time Spatial Predictor 🔮")
    st.write(
        "Move the sliders below to construct a custom flower profile. "
        "The model will immediately map its location in the space and determine its cluster classification."
    )
    
    col_pr1, col_pr2 = st.columns([1, 1.4])
    
    with col_pr1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Configure Sample Dimensions")
        
        input_length = st.slider("Input Petal Length (cm)", 1.0, 7.0, 3.5, 0.1)
        input_width = st.slider("Input Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict logic
        new_sample = np.array([[input_length, input_width]])
        if scaling_enabled:
            scaled_sample = scaler.transform(new_sample)
            predicted_c = kmeans_model.predict(scaled_sample)[0]
        else:
            predicted_c = kmeans_model.predict(new_sample)[0]
            
        # Radiant glowing box displaying result
        st.markdown(
            f'<div class="prediction-box">'
            f'<span>CLASSIFIED INTO</span>'
            f'<h2>CLUSTER {predicted_c}</h2>'
            f'<span>Nearest centroid distance updated</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    with col_pr2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Input Location Mapping")
        
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        apply_dark_theme(fig_pr, ax_pr, "Target Vector Alignment", "Petal Length (cm)", "Petal Width (cm)")
        
        # Plot clusters with dimmed alpha
        for cluster_id in range(k_val):
            subset = df_iris[df_iris['predicted_cluster'] == cluster_id]
            ax_pr.scatter(
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                color=NEON_COLORS[cluster_id % len(NEON_COLORS)],
                alpha=0.3,
                s=50,
                zorder=3
            )
            
        # Draw target vector (User input) as a giant glowing star
        ax_pr.scatter(
            input_length,
            input_width,
            color='#ffff00',
            edgecolor='#ffffff',
            marker='*',
            s=400,
            linewidths=1.5,
            label='Your Input Sample',
            zorder=12
        )
        
        # Centroids
        ax_pr.scatter(
            original_centroids[:, 0],
            original_centroids[:, 1],
            color='#ffffff',
            marker='X',
            s=180,
            edgecolor='#ff007f',
            label='Centroids',
            zorder=8
        )
        ax_pr.legend(facecolor='#07080f', edgecolor='#1e2030', labelcolor='#8f9cae')
        st.pyplot(fig_pr)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Tab 5: Evaluation Bench -----------------
with tab5:
    st.subheader("Ground-Truth Verification Bench")
    st.write(
        "Clustering is unsupervised, meaning the model works completely blind to the actual species label. "
        "Here, we match the cluster partitions against the original biological species to evaluate the quality of grouping."
    )
    
    col_ev1, col_ev2 = st.columns([1.2, 1])
    
    with col_ev1:
        st.markdown("### Alignment Matrix (Cross-Tabulation)")
        crosstab_df = pd.crosstab(df_iris['species'], df_iris['predicted_cluster'])
        st.dataframe(crosstab_df, use_container_width=True)
        
    with col_ev2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Partition Evaluation Details")
        
        if k_val == 3:
            st.info(
                "💡 **Cluster Assessment (k = 3):**\n\n"
                "- **Setosa** is perfectly separated from the other classes. "
                "The K-Means algorithm groups it with 100% biological accuracy.\n"
                "- **Versicolor** and **Virginica** have slightly overlapping petal attributes in nature, "
                "resulting in a small fraction of samples being clustered across boundaries. "
                "This is expected and verifies that K-Means successfully extracted the data's true structure!"
            )
        else:
            st.warning(
                f"💡 **Cluster Assessment (k = {k_val}):**\n\n"
                f"The biological dataset contains exactly **3 natural groups**. "
                f"Your selected cluster count ($k={k_val}$) divides the data into a different number of boundaries. "
                "This serves as a visual example of over-clustering or under-clustering. "
                "Set $k=3$ to inspect true biological mapping!"
            )
        st.markdown('</div>', unsafe_allow_html=True)
