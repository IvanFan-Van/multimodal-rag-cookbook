import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import transformers
from tqdm import tqdm
from pypdf import PdfReader
import html2text
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)

# Load the data
def load_breed_data():
    base_path = Path('breed_traits')
    
    # Load breed traits
    traits_df = pd.read_csv(base_path / 'breed_traits.csv')
    
    # Load trait descriptions
    trait_desc_df = pd.read_csv(base_path / 'trait_description.csv')
    
    # Create a comprehensive description for each breed
    breed_descriptions = []
    
    for _, breed_row in traits_df.iterrows():
        breed_name = breed_row['Breed']
        description_parts = [f"The {breed_name} has the following characteristics:"]
        
        for trait in trait_desc_df['Trait']:
            if trait in breed_row and trait in trait_desc_df['Trait'].values:
                trait_score = breed_row[trait]
                
                # Skip coat type and length as they are categorical
                if trait not in ['Coat Type', 'Coat Length']:
                    score_text = 'very low' if trait_score == 1 else \
                                'low' if trait_score == 2 else \
                                'moderate' if trait_score == 3 else \
                                'high' if trait_score == 4 else 'very high'
                    
                    description_parts.append(f"- {trait}: {score_text}")
                else:
                    description_parts.append(f"- {trait}: {trait_score}")
        
        breed_descriptions.append({
            'breed': breed_name,
            'description': '\n'.join(description_parts),
        })
        break
    
    return breed_descriptions

def convert_pdf_to_markdown(pdf_path):
    # 配置转换选项
    options = {
        'image_dir': './images',  # 图片保存目录
        'image_format': 'png',    # 图片格式
        'single_line': False,     # 是否将段落转换为单行
        'detect_tables': False,    # 是否检测表格
        'detect_lists': True      # 是否检测列表
    }
    
    # 使用 marker-pdf 转换
    markdown_content = convert_pdf(pdf_path, **options)
    
    return markdown_content

from pprint import pprint as pp
def main():
    # print("Loading breed data...")
    # breed_descriptions = load_breed_data()

    markdown_content = convert_pdf_to_markdown(r'breed_pdfs\affenpinscher.pdf')

    # print("Creating embeddings...")
    # embeddings = create_embeddings(breed_descriptions)
    
    # print("Saving database...")
    # save_database(embeddings)
    
    # print("Database creation completed!")

if __name__ == '__main__':
    main()