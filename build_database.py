# Create embeddings
def create_embeddings(breed_descriptions):
    # Initialize the sentence transformer model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create embeddings for each breed description
    embeddings = []
    for breed_data in breed_descriptions:
        embedding = model.encode(breed_data['description'])
        embeddings.append({
            'breed': breed_data['breed'],
            'embedding': embedding.tolist(),
            'description': breed_data['description'],
            'traits': breed_data['traits']
        })
    
    return embeddings

# Save the database
def save_database(embeddings, output_file='dog_breed_database.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
