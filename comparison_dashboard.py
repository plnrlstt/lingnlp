import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title('Language Taxonomy Comparison')

# Load data
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv('taxonomy_all7500_fullnames.csv')
        df1.columns = ['language', 'cluster_1']
    except FileNotFoundError:
        st.error('Error: `lang_tax_ALL.csv` not found.')
        return None, None

    try:
        # Use a robust method to read the potentially malformed lang2tax.txt
        with open('lang2tax.txt', 'r', encoding='latin1') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            try:
                lang, cluster = line.strip().rsplit(',', 1)
                data.append({'language': lang.lower(), 'cluster_2': int(cluster)})
            except ValueError:
                pass # Skip malformed lines
        df2 = pd.DataFrame(data)

    except FileNotFoundError:
        st.error('Error: `lang2tax.txt` not found.')
        return None, None

    return df1, df2

df1, df2 = load_data()

if df1 is not None and df2 is not None:
    # Merge dataframes
    merged_df = pd.merge(df1, df2, on='language', how='outer')

    # Identify changes
    merged_df['cluster_changed'] = merged_df['cluster_1'].notna() & merged_df['cluster_2'].notna() & (merged_df['cluster_1'] != merged_df['cluster_2'])

    # --- Dashboard Layout ---
    st.header("Taxonomy Comparison Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Languages in Taxonomy 1", len(df1))
    col2.metric("Languages in Taxonomy 2", len(df2))
    col3.metric("Common Languages", len(merged_df.dropna(subset=['cluster_1', 'cluster_2'])))
    col4.metric("Languages with Changed Clusters", len(merged_df[merged_df['cluster_changed']]))

    st.header("Cluster Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taxonomy 1 (taxonomy_all7500_fullnames.csv)")
        cluster1_counts = df1['cluster_1'].value_counts().reset_index()
        cluster1_counts.columns = ['Cluster', 'Count']
        fig1 = px.bar(cluster1_counts, x='Cluster', y='Count', title='Languages per Cluster in Taxonomy 1')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Taxonomy 2 (lang2tax.txt)")
        cluster2_counts = df2['cluster_2'].value_counts().reset_index()
        cluster2_counts.columns = ['Cluster', 'Count']
        fig2 = px.bar(cluster2_counts, x='Cluster', y='Count', title='Languages per Cluster in Taxonomy 2')
        st.plotly_chart(fig2, use_container_width=True)

    st.header("Languages with Changed Clusters")
    changed_languages_df = merged_df[merged_df['cluster_changed']][['language', 'cluster_1', 'cluster_2']]
    st.dataframe(changed_languages_df)
