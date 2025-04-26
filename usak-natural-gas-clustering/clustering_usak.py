from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import folium

class ClusteringUsak:
    def __init__(self):
        self.df = pd.read_excel('01-Clustering Usak Natural Gas Consumption.xlsx')
        self.df_sample = self.df.sample(n= 1000, random_state= 42)
        self.X_sample = None
        self.X =None

    def find_n(self):
        correlation_matrix = self.df.corr()

        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

        info_prop = eigenvalues / eigenvalues.sum()

        cum_info_prop = np.cumsum(info_prop)

        fig1, ax1 = plt.subplots()
        ax1.plot(np.arange(1, len(info_prop)+ 1), cum_info_prop, 'bo-')
        ax1.hlines(y = 95, xmin = 0, xmax = 15)
        ax1.vlines(x = 4, ymin= 0 ,ymax=1)
        st.pyplot(fig1)
        fig1.savefig("n1.png", bbox_inches='tight')
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        n = np.argmax(cum_info_prop >= 0.95) + 1 
        ax2.plot(cum_info_prop, 'bo-', linewidth=2)
        ax2.hlines(y=.95, xmin=0, xmax=len(cum_info_prop)-1, color='r', linestyle='--')  
        ax2.vlines(x=n-1, ymin=0, ymax=1, color='g', linestyle='--')
        ax2.set_title('Cumulative Information Percentages')
        ax2.set_xlabel('Principal Axes')
        ax2.set_xticks(np.arange(1, len(cum_info_prop) + 1))  
        ax2.set_ylabel('Cumulative Proportion of Variance Explained')
        ax2.grid(True)
        st.pyplot(fig2)

        fig2.savefig("n2.png", bbox_inches='tight')
        plt.close(fig2)
       
    def preprocessing(self, X):
           
        imputer = SimpleImputer(strategy="mean")  
        X = imputer.fit_transform(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X) 
        return X

    def KMeansFit_predict(self,X_pca,filter_name):
        #from sklearn.metrics import silhouette_score
        kmeans = KMeans(n_clusters=4 , random_state= 42)
        clusters = kmeans.fit_predict(X_pca)
        #score = silhouette_score(X_pca, kmeans.labels_)
        #print(f'Score -> {score}')
        self.df["cluster"] = clusters
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis") 
        ax.set_title("KMeans Clustering (with PCA)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="Cluster") 
        st.pyplot(fig)
        fig.savefig(f"kmeans_model_{filter_name}.png", bbox_inches='tight')
        plt.close(fig)


    def AggFit_predict(self, sample_df, X_pca, filter_name):
        
        agg = AgglomerativeClustering(n_clusters= 4)
        agg_cluster = agg.fit_predict(X_pca)
        sample_df["agg_cluster"] = agg_cluster

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_cluster, cmap="viridis")  
        ax.set_title("Agg Clustering (with PCA)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="Cluster")  
        st.pyplot(fig)

        fig.savefig(f"agg_model_{filter_name}.png", bbox_inches='tight')
        plt.close(fig)


    def dbscan_fit_predict(self, sample_df, X_pca, filter_name):

        dbscan = DBSCAN(eps= 0.5, min_samples=5)
        dbscan_cluster = dbscan.fit_predict(X_pca)
        sample_df["dbscan_cluster"] = dbscan_cluster
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_cluster, cmap="viridis")
        ax.set_title("DBSCAN Clustering (with PCA)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
        fig.savefig(f"dbscan_model_{filter_name}.png", bbox_inches='tight')
        plt.close(fig)

    def calculate_Consumption_Pattern_Based_Clustering(self):
        
        self.X = self.df.loc[:, "Jan":"Dec"] 
        self.X_sample = self.df_sample.loc[:, "Jan":"Dec"] 
        self.X = self.preprocessing(self.X)
        self.X_sample = self.preprocessing(self.X_sample)

        pca = PCA(n_components=11)
        X_pca = pca.fit_transform(self.X)
        X_pca_sample = pca.transform(self.X_sample)

        self.KMeansFit_predict(X_pca, "consumption_pattern_based")
        self.AggFit_predict(self.df_sample, X_pca_sample, "consumption_pattern_based")
        self.dbscan_fit_predict(self.df_sample, X_pca_sample, "consumption_pattern_based")

    def calculate_Clustering_Based_on_Exchange_Rate_Influence(self):
        self.X = self.df.loc[:, "Jun usd/tr" : "Dec"].drop(columns=["longitude", "latitude"])
        self.X_sample = self.df_sample.loc[:, "Jun usd/tr" : "Dec"].drop(columns=["longitude", "latitude"])
        self.X = self.preprocessing(self.X)
        self.X_sample = self.preprocessing(self.X_sample)

        #PCA
        pca = PCA(n_components=11)
        X_pca = pca.fit_transform(self.X)
        X_pca_sample = pca.transform(self.X_sample)

        self.KMeansFit_predict(X_pca, "clustering_based_on_exchange_rate")
        self.dbscan_fit_predict(self.df_sample, X_pca_sample, "clustering_based_on_exchange_rate")

    def find_outliers(self):
        self.df = self.df.reset_index(drop=True)

        self.X = self.df.loc[:, "Jan":"Dec"]
        print("self.df index:", self.df.index)

        self.X = self.preprocessing(self.X)

        pca = PCA(n_components=11)
        X_pca = pca.fit_transform(self.X)
        print("X_pca shape:", X_pca.shape)

        z_scores = np.abs(zscore(X_pca))
        outliers = (z_scores > 1.0).any(axis=1)

        print("Bulunan outlier sayısı:", outliers.sum()) 

        self.df["anomaly_z"] = outliers
        self.df = self.df[self.df["anomaly_z"] == False]

        print("Temizlenen veri boyutu:", self.df.shape)


        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.scatter(X_pca[outliers, 0], X_pca[outliers, 1], c="red", label="Outliers", alpha=0.5)
        ax.scatter(X_pca[~outliers, 0], X_pca[~outliers, 1], c="blue", label="Normal", alpha=0.7)
        ax.set_title("Outlier Detection (PCA 2D Projection)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        fig.savefig("outliers.png", bbox_inches='tight')
        plt.close(fig)

    
