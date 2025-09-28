import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

def format_speakers(count):
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    else:
        return f"{int(count):,}"

st.set_page_config(layout="wide")

st.title('Language Resource Distribution Analysis (with all languages)')

# Load data
try:
    df = pd.read_csv('resource_distribution_with_speakers_and_all_languages.csv', delimiter=';')
    df['numberOfSpeakers'] = pd.to_numeric(df['numberOfSpeakers'], errors='coerce').fillna(0)

    # Load taxonomy data for full language names
    try:
        taxonomy_df = pd.read_csv('taxonomy_no_lang_id.csv', delimiter=';')
        df = pd.merge(df, taxonomy_df[['Full Language Name']], left_on='LANGUAGE', right_on='Full Language Name', how='left')
        df['Full Language Name'] = df['Full Language Name'].fillna(df['LANGUAGE']) # Use original LANGUAGE if full name not found
    except FileNotFoundError:
        st.warning('Warning: `taxonomy_no_lang_id.csv` not found. Full language names will not be available.')
        df['Full Language Name'] = df['LANGUAGE'] # Fallback to short name

except FileNotFoundError:
    st.error('Error: `resource_distribution_with_speakers_and_all_languages.csv` not found.')
    st.stop()

# Prepare data for clustering using log base 10
df['log_unlabelled'] = np.log10(df['UNLABELLED RESOURCE COUNT'].replace(0, 1))
df['log_labelled'] = np.log10(df['LABELLED RESOURCE COUNT'].replace(0, 1))

# K-Means Clustering
X = df[['log_unlabelled', 'log_labelled']]
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Reorder clusters
centers = kmeans.cluster_centers_
center_sums = centers.sum(axis=1)
sorted_indices = np.argsort(center_sums)
label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted_indices)}
df['cluster'] = df['cluster'].map(label_mapping)
sorted_centers = centers[sorted_indices]

# Aggregate data by cluster
cluster_agg = df.groupby('cluster').agg(
    language_count=('LANGUAGE', 'size'),
    total_speakers=('numberOfSpeakers', 'sum')
).reset_index()

cluster_agg['log_unlabelled'] = [center[0] for center in sorted_centers]
cluster_agg['log_labelled'] = [center[1] for center in sorted_centers]

# Define cluster colors
cluster_colors = ['#6A5ACD', '#483D8B', '#6495ED', '#90EE90', '#FFD700', '#FFA07A', '#CD5C5C']

# --- Visualizations ---

# Clustering Plot
st.header("Language Resource Clustering")
fig_cluster = go.Figure()

# Add individual language points
fig_cluster.add_trace(go.Scatter(
    x=df['log_unlabelled'],
    y=df['log_labelled'],
    mode='markers',
    marker=dict(color='lightgrey', size=5),
    hoverinfo='text',
    text=df['Full Language Name'].fillna(df['LANGUAGE']),
    name='Languages'
))

# Add smoothed polygons
for i in range(6):
    cluster_points = df[df['cluster'] == i][['log_unlabelled', 'log_labelled']].values
    if len(cluster_points) > 2:
        try:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            tck, u = splprep([hull_points[:,0], hull_points[:,1]], s=0, per=True)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            fig_cluster.add_trace(go.Scatter(
                x=x_new,
                y=y_new,
                mode='lines',
                line=dict(color=cluster_colors[i], dash='dash'),
                fill='toself',
                fillcolor=f'rgba({int(cluster_colors[i][1:3], 16)}, {int(cluster_colors[i][3:5], 16)}, {int(cluster_colors[i][5:7], 16)}, 0.3)',
                opacity=0.3,
                showlegend=False,
                hoverinfo='none'
            ))
        except Exception:
            pass

# Add cluster centers
fig_cluster.add_trace(go.Scatter(
    x=cluster_agg['log_unlabelled'],
    y=cluster_agg['log_labelled'],
    mode='markers',
    marker=dict(
        size=cluster_agg['language_count'],
        sizemode='area',
        sizemin=4,
        sizeref=max(cluster_agg['language_count'])/(60.**2),
        color=[cluster_colors[c] for c in cluster_agg['cluster']],
        showscale=False
    ),
    hoverinfo='text',
    text=[f"Cluster {i}<br>Languages: {lc}<br>Speakers: {format_speakers(ts)}" for i, lc, ts in zip(cluster_agg.index, cluster_agg['language_count'], cluster_agg['total_speakers'])],
    name='Clusters'
))

fig_cluster.update_layout(
    title='Language Resource Distribution with Clustering',
    xaxis_title='Log of Unlabelled Resource Count',
    yaxis_title='Log of Labelled Resource Count',
    showlegend=False,
    width=800,
    height=600
)
st.plotly_chart(fig_cluster, use_container_width=True)





st.header("Data")
st.dataframe(df)
