import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
from clustering_usak import ClusteringUsak
import matplotlib.pyplot as plt
from PIL import Image
import os

class Main:
    def __init__(self):
        self.instance = ClusteringUsak()

    def show_images(self, image_files):
        for img_path in image_files:
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption=img_path, use_container_width=True) 


    def main(self):
        st.sidebar.title("Menu")
        page = st.sidebar.radio(
            "Sayfa Seçin",
            ["Menu", 
             "📊 Calculate Consumption Pattern Based Clustering", 
             "📊 Calculate Clustering Based on Exchange Rate Influence", 
             "Find Outliers", 
             ]
        )

        if page == "Menu":
            st.title("📌 Project Overview: Natural Gas Consumption Clustering")
            st.markdown("""
            ### 📊 Project Description:
            This project performs **clustering analysis** on natural gas consumption data 
            collected from the Uşak region.

            The main goals are:
            - Identify **consumption patterns** using clustering algorithms (KMeans, Agglomerative Clustering, DBSCAN).
            - Analyze the **impact of currency exchange rates** on consumption behaviors.
            - Detect **anomalies and outliers** in the consumption data.
            - Perform **geospatial profiling** to visualize customer distribution on the map.

            The dataset includes **monthly consumption values**, **exchange rates**, and **geolocation information**.

            #### 🧪 Techniques Used:
            - **Principal Component Analysis (PCA)** for dimensionality reduction.
            - **KMeans**, **Agglomerative Clustering**, **DBSCAN** for unsupervised clustering.
            - **Z-Score Method** for anomaly detection.
            - **Folium** for interactive geospatial visualization.
            """)


        elif page == "📊 Calculate Consumption Pattern Based Clustering":
            st.title("Graphs")
            
            #self.instance.calculate_Consumption_Pattern_Based_Clustering()

            
            st.subheader("Clustering Results (Consumption Pattern Based)")
            image_files = [
                "./plots/kmeans_model_consumption_pattern_based.png",
                "./plots/agg_model_consumption_pattern_based.png",
                "./plots/dbscan_model_consumption_pattern_based.png"
            ]
            self.show_images(image_files)

        elif page == "📊 Calculate Clustering Based on Exchange Rate Influence":
            st.title("Graphs")
            
            #self.instance.calculate_Clustering_Based_on_Exchange_Rate_Influence()

            
            st.subheader("Clustering Results (Exchange Rate Influence)")
            image_files = [
                "./plots/kmeans_model_clustering_based_on_exchange_rate.png",
                "./plots/dbscan_model_clustering_based_on_exchange_rate.png"
            ]
            self.show_images(image_files)

        elif page == "Find Outliers":
            st.title("Graphs")
            #self.instance.find_outliers()

            # 🎯 Outlier plotu:
            st.subheader("Outlier Detection Results")
            image_files = ["./plots/outliers.png"]
            self.show_images(image_files)

            


if __name__ == "__main__":
    x = Main()
    x.main()
