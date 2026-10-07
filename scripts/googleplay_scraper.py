import pandas as pd
import time
import random
import os
from google_play_scraper import Sort, reviews
from typing import List

def robust_fetch_and_save(
    app_ids: List[str],
    output_file: str = "massive_raw_reviews.csv",
    langs: List[str] = ['ky', 'ru'],
    country: str = 'kg',
    count_per_request: int = 10000
):
    """
    Fault-tolerant parser. Saves data on the fly, avoids bans, and ignores errors.
    """
    sort_types = [Sort.NEWEST, Sort.MOST_RELEVANT]
    
    file_exists = os.path.isfile(output_file)
    
    for app_id in app_ids:
        print(f"\n[{time.strftime('%X')}] Starting collection for: {app_id}")
        app_reviews = []
        
        for lang in langs:
            for sort_type in sort_types:
                print(f"  -> Request: Language '{lang}', Sort by '{sort_type.name}'...")
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result, _ = reviews(
                            app_id,
                            lang=lang,
                            country=country,
                            sort=sort_type,
                            count=count_per_request
                        )
                        
                        for review in result:
                            review['app_id'] = app_id
                            review['os_locale'] = lang
                            app_reviews.append(review)
                            
                        break 
                        
                    except Exception as e:
                        print(f"     [ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        if attempt < max_retries - 1:
                            sleep_time = random.uniform(10, 20)
                            print(f"     Waiting {sleep_time:.1f} sec. before retrying...")
                            time.sleep(sleep_time)
                        else:
                            print("     [SKIP] Failed to get data. Moving on.")
                
                time.sleep(random.uniform(2.0, 5.0))
        
        if app_reviews:
            df = pd.DataFrame(app_reviews)
            df_clean = df.drop_duplicates(subset=['reviewId']).reset_index(drop=True)
            df_clean = df_clean[['reviewId', 'userName', 'content', 'score', 'at', 'app_id', 'os_locale']]
            
            df_clean.to_csv(
                output_file, 
                mode='a', 
                header=not file_exists, 
                index=False, 
                encoding='utf-8'
            )
            file_exists = True 
            print(f"[SUCCESS] +{len(df_clean)} unique reviews added to the database.")
        else:
            print(f"[EMPTY] No data for {app_id}")
            
        big_sleep = random.uniform(15.0, 30.0)
        print(f"Cooling down for {big_sleep:.1f} sec. before the next app...")
        time.sleep(big_sleep)

if __name__ == "__main__":
    with open("applications.txt", "r", encoding="utf-8") as f:
        apps = [line.strip() for line in f if line.strip()]
    
    robust_fetch_and_save(app_ids=apps, output_file="final_raw_database.csv")
    print("\nData collection is complete. Time for coffee!")