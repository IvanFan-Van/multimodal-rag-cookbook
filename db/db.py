import kagglehub
import pandas as pd
import os
import requests
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModel
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download breed summary metrics
path = kagglehub.dataset_download("sujaykapadnis/dog-breeds")
print("Path to dataset files:", path)

# Download dataset
ds = load_dataset("ajinkyakolhe112/dog_breed_classification_kaggle")

# Load the data
breed_traits = pd.read_csv(os.path.join(path, "breed_traits.csv"))
breed_rank = pd.read_csv(os.path.join(path, "breed_rank.csv"))
breed_traits_long = pd.read_csv(os.path.join(path, "breed_traits_long.csv"))
trait_description = pd.read_csv(os.path.join(path, "trait_description.csv"))

# Download breed PDFs
links = breed_rank["links"].tolist()
pdf_base_url = "https://images.akc.org/pdf/breeds/standards/"

pdf_urls = []
breed_names = []
for link in links:
    breed_name = link.rstrip("/").split("/")[-1]
    breed_names.append(breed_name)
    words = [word.capitalize() for word in breed_name.split("-")]
    breed_name = "".join(words)
    pdf_urls.append(os.path.join(pdf_base_url, breed_name + ".pdf"))

def download_breed_pdf(pdf_url, breed_name, pdf_dir=None):
    if pdf_dir is None:
        pdf_dir = Path("breed_pdfs")
    pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir / f"{breed_name}.pdf"

    try:
        # 下载PDF文件
        response = requests.get(pdf_url, stream=True, verify=False)
        response.raise_for_status()  # 检查是否下载成功
        
        # 保存文件
        with open(pdf_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_file.write(chunk)
        
        print(f"成功下载PDF文件: {pdf_path}")
        return str(pdf_path)
    
    except requests.exceptions.RequestException as e:
        print(f"下载PDF时发生错误: {e}")
    return None

pdf_dir = "./breed_pdfs"
for url, name in zip(pdf_urls, breed_names):
    download_breed_pdf(url, name, pdf_dir)

# Initialize embedding model
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
truncate_dim = 512

def extract_image_features(images):
    image_features = model.encode_image(images, truncate_dim=truncate_dim)
    return image_features

def extract_text_features(texts):
    text_features = model.encode_text(texts, truncate_dim=truncate_dim)
    return text_features

# === build Image Embedding dataset ===
image_features_list = []
for img in tqdm(ds['train'][:100]['image']):
    features = extract_image_features(img)
    image_features_list.append(features)

image_features = np.vstack(image_features_list)
df = pd.DataFrame(image_features)
df.to_parquet(r"./db/image_features.parquet")

# === build text embedding dataset ===
# context chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
markdown_dir = Path(r"d:/HKU/Inno Wing RA/multimodal-rag-tutorial/md_outputs")
documents = []
for path in markdown_dir.rglob("*.md"):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        documents.append(content)

chunks = splitter.split_text("\n\n".join(documents))

text_features_list = []
for doc in tqdm(chunks):
    features = extract_text_features(doc)
    text_features_list.append(features)

text_features = np.vstack(text_features_list)
df = pd.DataFrame(text_features)
df.to_parquet(r"./db/text_features.parquet")

# build text chunk dataset
df = pd.DataFrame(chunks)
df.to_parquet(r"./db/text_chunks.parquet")

print("Done!")