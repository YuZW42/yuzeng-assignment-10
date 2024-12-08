from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Initialize CLIP model and load embeddings
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# load the embeddings
df = pd.read_pickle('image_embeddings.pickle')
embeddings_tensor = torch.tensor(df['embedding'].tolist())

pca = None

def get_pca_embeddings(k, embeddings):
    global pca
    pca = PCA(n_components=k)
    return pca.fit_transform(embeddings)

def transform_query_with_pca(query_embedding):
    global pca
    return pca.transform(query_embedding.reshape(1, -1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query_type = request.form['query_type']
        use_pca = 'use_pca' in request.form
        k_components = int(request.form.get('k_components', 50)) if use_pca else None

        # apply PCA if requested
        embeddings = embeddings_tensor.numpy()
        if use_pca:
            app.logger.info(f"Using PCA with {k_components} components")
            embeddings = get_pca_embeddings(k_components, embeddings)

        if query_type == 'text':
            text = request.form['text']
            text_input = tokenizer([text])
            with torch.no_grad():
                query_embedding = model.encode_text(text_input).numpy()
                if use_pca:
                    query_embedding = transform_query_with_pca(query_embedding)
                else:
                    query_embedding = F.normalize(torch.from_numpy(query_embedding)).numpy()

        elif query_type == 'image':
            if 'image' not in request.files:
                return 'No image uploaded', 400
            file = request.files['image']
            img = Image.open(file)
            image_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                query_embedding = model.encode_image(image_tensor).numpy()
                if use_pca:
                    query_embedding = transform_query_with_pca(query_embedding)
                else:
                    query_embedding = F.normalize(torch.from_numpy(query_embedding)).numpy()
        else:  
        
            file = request.files['image']
            text = request.form['text']
            weight = float(request.form.get('weight', 0.8))
            
            img = Image.open(file)
            image_tensor = preprocess(img).unsqueeze(0)
            text_input = tokenizer([text])
            
            with torch.no_grad():
                image_query = model.encode_image(image_tensor).numpy()
                text_query = model.encode_text(text_input).numpy()
                
                if use_pca:
                    query_embedding = weight * transform_query_with_pca(text_query) + \
                                    (1.0 - weight) * transform_query_with_pca(image_query)
                else:
                    image_query = F.normalize(torch.from_numpy(image_query)).numpy()
                    text_query = F.normalize(torch.from_numpy(text_query)).numpy()
                    query_embedding = weight * text_query + (1.0 - weight) * image_query
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calculate distances and get top results, from lab
        distances = euclidean_distances(query_embedding, embeddings).flatten()
        top_indices = np.argsort(distances)[:5]

        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'filename': df.iloc[idx]['file_name'],
                'similarity': 1 / (1 + distances[idx])  #  distance --> similarity score
            })

        return render_template('results.html', results=results)

    except Exception as e:
        app.logger.error(f"Error during search: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)

if __name__ == '__main__':
    app.run(debug=True, port=3000)