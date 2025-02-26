import torch
import pandas as pd
import numpy as np
from PIL import Image

class Retriever:
    def __init__(self, model, ds):
        self.model = model
        self.truncate_dim = 512
        self.image_features = pd.read_parquet(r"./db/image_features.parquet").to_numpy()
        self.text_features = pd.read_parquet(r"./db/text_features.parquet").to_numpy()
        self.text_chunks = pd.read_parquet(r"./db/text_chunks.parquet").to_numpy()
        self.class_labels = ds.features['label'].names
        self.ds = ds
        
    def extract_image_features(self, images):
        image_features = self.model.encode_image(images, truncate_dim=self.truncate_dim).reshape(1, -1)
        return image_features

    def extract_text_features(self, texts):
        text_features = self.model.encode_text(texts, truncate_dim=self.truncate_dim).reshape(1, -1)
        return text_features

    def combine_features(self, features_list, weights=None):
        """
        Combine multiple features using weighted average.
        Parameters:
            features_list (list): List of features to combine.
            weights (list): List of weights for each feature. Defaults to None.
        Returns:
            numpy.ndarray: Combined features.
        """
        if weights is None:
            weights = [1.0] * len(features_list)
        weighted_features = [f * w for f, w in zip(features_list, weights)]
        combined_features = np.sum(weighted_features, axis=0)
        return combined_features / np.linalg.norm(combined_features)

    def retrieve(self, queries, weights=None, top_k=5):
        if not isinstance(queries, list):
            queries = [queries]
        features_list = []
        for query in queries:
            if isinstance(query, str):
                query_features = self.extract_text_features(query)
            elif isinstance(query, Image.Image):
                query_features = self.extract_image_features(query)
            else:
                raise ValueError("Invalid query. Please pass in string or PIL.Image")
            features_list.append(query_features)

        combined_features = self.combine_features(features_list, weights)
        combined_norm = np.linalg.norm(combined_features, axis=1, keepdims=True)
        
        # retrieve images
        image_norm = np.linalg.norm(self.image_features, axis=1, keepdims=True)
        similarities = np.dot(combined_features, self.image_features.T) / (combined_norm * image_norm.T)
        
        # retrieve documents
        text_norm = np.linalg.norm(self.text_features, axis=1, keepdims=True)
        similarities_text = np.dot(combined_features, self.text_features.T) / (combined_norm * text_norm.T)

        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        results = self.ds[top_indices]
        images, labels = results['image'], results['label']
        names = [self.class_labels[i] for i in labels]
        image_similarity_scores = similarities[0][top_indices]

        top_indices_text = np.argsort(similarities_text[0])[-top_k:][::-1]
        results_text = self.text_chunks[top_indices_text]
        text_similarity_scores = similarities_text[0][top_indices_text]
        return {
            "images": images, 
            "image_similarity_scores": image_similarity_scores,
            "labels": labels, 
            "names": names, 
            "docs": results_text,
            "text_similarity_scores": text_similarity_scores
        }
