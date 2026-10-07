import pandas as pd
import os
import sys
import fasttext
from tqdm import tqdm

MODEL_PATH = 'lid.176.ftz'

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File '{MODEL_PATH}' not found!")
    sys.exit(1)

fasttext.FastText.eprint = lambda x: None
lang_model = fasttext.load_model(MODEL_PATH)

def process_chunk_lenient_no_kazakh(chunk: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df_clean = chunk.dropna(subset=[text_col]).copy()
    df_clean[text_col] = df_clean[text_col].astype(str)
    
    has_cyrillic = df_clean[text_col].str.contains(r'[а-яА-ЯөүңӨҮҢ]', regex=True, na=False)
    df_cyr = df_clean[has_cyrillic].copy()
    
    if df_cyr.empty:
        return pd.DataFrame()
        
    text_series_lower = df_cyr[text_col].str.lower()
    
    kk_chars = df_cyr[text_col].str.contains(r'[әғқұіһӘҒҚҰІҺ]', regex=True, na=False)
    
    kk_words = text_series_lower.str.contains(
        r'\b(?:жақсы|жоқ|өте|үшін|рахмет|бағдарлама|қосымша|қалай|бірақ|енді|менің|бұл|осы)\b', 
        regex=True, na=False
    )
    
    is_kazakh = kk_chars | kk_words
    
    df_no_kazakh = df_cyr[~is_kazakh].copy()
    
    if df_no_kazakh.empty:
        return pd.DataFrame()
        
    text_series_no_kk = df_no_kazakh[text_col].str.lower()

    ky_chars = df_no_kazakh[text_col].str.contains(r'[ңөүҢӨҮ]', regex=True, na=False)
    ky_words = text_series_no_kk.str.contains(
        r'\b(?:жана|бул|менен|үчүн|жакшы|сонун|эмес|жок|рахмат|тиркеме|болот|керек|эмне|бирок|ушул)\b', 
        regex=True, na=False
    )
    golden_mask = ky_chars | ky_words
    df_approved_instantly = df_no_kazakh[golden_mask].copy()
    
    df_uncertain = df_no_kazakh[~golden_mask].copy()
    
    if not df_uncertain.empty:
        texts_for_ft = df_uncertain[text_col].str.replace('\n', ' ').tolist()
        predictions, confidences = lang_model.predict(texts_for_ft)
        
        is_ky_list = []
        for i in range(len(predictions)):
            lang = predictions[i][0].replace('__label__', '')
            conf = confidences[i][0]
            
            if lang == 'kk':
                is_ky_list.append(False)
            else:
                is_ky_list.append(lang == 'ky' and conf >= 0.40)
            
        df_approved_by_ml = df_uncertain[is_ky_list].copy()
    else:
        df_approved_by_ml = pd.DataFrame()
        
    final_chunk = pd.concat([df_approved_instantly, df_approved_by_ml])
    
    return final_chunk

def filter_massive_dataset(input_file: str, output_file: str, text_col: str = 'content', chunk_size: int = 50000):
    print(f"Starting filter (Mixed language: ON | Kazakh language: BLOCKED). File: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f) - 1 
    total_chunks = (total_lines // chunk_size) + 1
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    total_ky_reviews = 0
    file_exists = False
    
    chunk_iterator = pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8')
    
    for chunk in tqdm(chunk_iterator, total=total_chunks, desc="Filtering"):
        filtered_chunk = process_chunk_lenient_no_kazakh(chunk, text_col=text_col)
        
        if not filtered_chunk.empty:
            total_ky_reviews += len(filtered_chunk)
            filtered_chunk.to_csv(output_file, mode='a', header=not file_exists, index=False, encoding='utf-8')
            file_exists = True

    print("\n" + "="*45)
    print("FILTERING COMPLETED!")
    print(f"Remaining rows (Kyrgyz + Mixed language without Kazakh): {total_ky_reviews}")
    print(f"Dataset saved to: {output_file}")
    print("="*45)

if __name__ == "__main__":
    
    INPUT_FILE = "final_raw_database.csv"  
    OUTPUT_FILE = "kyrgyz_only_lenient.csv" 
    
    filter_massive_dataset(INPUT_FILE, OUTPUT_FILE, text_col="content", chunk_size=50000)