"""
Data loading and preprocessing module for Sentiment Analysis.
Provides robust text cleaning, seed data generation, and train/test splitting.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_text(text: str) -> str:
    """Clean and normalize raw text for sentiment classification.
    
    Operations performed:
    - HTML tag removal
    - URL removal
    - Email address removal
    - Non-alphanumeric cleanup (preserving basic punctuation)
    - Lowercase conversion
    - Whitespace normalization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 2. Remove URLs
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    
    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    
    # 4. Remove unwanted symbols but keep word chars and sentiment-bearing punctuation
    text = re.sub(r"[^a-zA-Z0-9\s!?.,'\"]", " ", text)
    
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text.lower()


def get_seed_dataset() -> pd.DataFrame:
    """Generate a curated, balanced seed dataset of sentiment reviews.
    
    Provides rich vocabulary and repeated sentiment indicators across movies,
    products, and services to ensure reliable model training and high accuracy.
    """
    positives = [
        "This movie is an absolute masterpiece! Brilliant acting, stunning direction, and great storytelling.",
        "I loved every single moment of this film. Highly recommended to everyone who loves good cinema.",
        "Outstanding performances, magnificent visuals, and an unforgettable, touching story. Truly great!",
        "One of the finest films ever made in contemporary cinema. Pure genius, highly recommended!",
        "A heartwarming and genuinely delightful experience from beginning to end. Loved it so much!",
        "The cinematography and score were breathtaking. Truly well executed and wonderfully inspiring.",
        "Incredible storytelling with immense emotional depth. A fantastic 10/10 work of art.",
        "Superb quality, arrived faster than expected and works like a charm. Extremely satisfied!",
        "Customer service was extraordinarily helpful and resolved my issue immediately. Great support!",
        "Fantastic product! Sleek design, high performance, and worth every dollar. Very happy!",
        "An uplifting, inspirational, and wonderfully crafted adventure. Really great movie!",
        "Exceptional talent displayed by the entire ensemble cast. Great direction and lovely acting.",
        "I was completely captivated by the suspense and wonderful character arcs. Highly recommended.",
        "Hands down the best purchase I have made all year. Great quality and highly satisfied!",
        "Brilliantly written script with clever dialogue, great humor, and heartwarming moments.",
        "Exceeded all my expectations. Flawless execution, remarkable pace, and truly great acting.",
        "A stunning visual spectacle accompanied by a deeply moving and beautiful soundtrack. Loved it.",
        "Very smooth experience, elegant interface, fast performance, and reliable support.",
        "Refreshing, authentic, and emotionally resonant. A great triumph and wonderful accomplishment!",
        "Five stars without hesitation. Excellent product, great value, and will definitely buy again.",
        "Loved the ending! Perfect wrap up to a fantastic series. Truly wonderful and satisfying.",
        "Remarkable attention to detail and high craftsmanship. Very impressed with the great quality.",
        "Such an energetic and positive vibe! Kept me smiling all day. Fantastic performance!",
        "The acting was top notch and the plot twists kept me entertained throughout. Great film.",
        "Great value for money, durable material, great design, and top-tier usability.",
        "Phenomenal direction. The director knows how to construct intense emotion and great suspense.",
        "An absolute joy to watch. Inspiring, uplifting, and beautifully shot. Best film this year.",
        "Top tier performance, responsive feedback, and overall supreme quality. Highly impressed!",
        "A memorable cinematic milestone with unforgettable character delivery and excellent screenplay.",
        "Fast shipping, friendly customer care, and excellent product condition. Great experience!",
        "Simply breathtaking. The screenplay deserves prestigious accolades. Fantastic writing.",
        "Wonderfully balanced humor with touching dramatic moments. Loved the whole story.",
        "A classic that stands the test of time. Mesmerizing, profound, and deeply enjoyable.",
        "The best sequel imaginable. Surpassed the original in every conceivable way. Great job!",
        "Super satisfied! Everything functioned straight out of the box. Highly recommended!",
        "Terrific experience! Quick delivery, great packaging, and the product is super high quality.",
        "Amazing performance and very easy to use. I love everything about this!",
        "Spectacular visuals and a very gripping plot. Truly a fantastic piece of cinema.",
        "The best customer support team I have ever dealt with. Polite, quick, and very effective.",
        "Delightful from start to finish. Hilarious jokes and great emotional moments.",
        "Excellent build quality and very comfortable. Five stars all the way!",
        "What a brilliant movie! Emotional, powerful, and deeply moving.",
        "Super happy with this item! Works like a charm and looks fantastic.",
        "Impressive speed, seamless user experience, and very reliable service.",
        "A true masterpiece of storytelling. Highly engaging and wonderfully executed.",
        "The battery life is incredible and the display is gorgeous. Best gadget purchase!",
        "Warm, charming, and thoroughly entertaining. A must watch for everyone!",
        "Great customer service, fast replacement, and friendly communication.",
        "Outstanding value, easy setup, and very durable materials.",
        "I am so impressed with how well this works. Definitely worth every penny!",
        "The acting was superb and the music was enchanting. Loved every minute.",
        "A wonderful experience that brightened my mood completely. Highly recommend!",
        "Flawless performance, zero lag, and very responsive. Excellent product!",
        "An astonishing achievement in filmmaking. Visually gorgeous and emotionally profound.",
        "Super clean design, intuitive UI, and very helpful documentation. Great job!",
        "Brilliant, poignant, and expertly directed. One of the best movies of the decade.",
        "Exceptional build and beautiful aesthetic. Exceeded my high expectations!",
        "So glad I bought this! Top notch build, great reliability, and fast shipping.",
        "A truly remarkable film with superb cinematography and fantastic acting.",
        "Best tool I have used for this job. Saved me hours of time and works flawlessly!"
    ]

    negatives = [
        "Terrible movie. Boring plot, dreadful acting, and a complete waste of time.",
        "I hated this film. Predictable story, awful pacing, and utterly wooden performances.",
        "One of the worst experiences I have ever sat through. Avoid at all costs, terrible film.",
        "Horrible direction, nonsensical plot holes, and obnoxious characters. Awful waste.",
        "Extremely disappointed. Shoddy build quality, terrible materials, and broke within two days.",
        "Dull, uninspiring, and drags on for far too long without any purpose. Very boring.",
        "Awful customer support. They refused to refund, were rude, and ignored my complaints.",
        "The pacing was painfully slow and the dialogue was cringe-worthy and terrible.",
        "Cheap material, misleading product description, and completely defective upon arrival.",
        "Don't waste your hard earned money on this disaster. Absolute garbage and terrible quality.",
        "Flat characters with zero chemistry. I could not care less about anyone. Very bad movie.",
        "Total letdown. The trailer made it look thrilling, but the actual movie is boring and bad.",
        "Unbearably loud and chaotic with zero coherent narrative. Just an awful mess.",
        "The app crashes constantly, freezes all the time, and drains battery life horribly.",
        "Poorly edited, confusing transitions, and awful audio mixing throughout. Very bad.",
        "Worst service I have ever dealt with. Rude staff, delayed orders, and zero communication.",
        "A frustrating waste of talent. Even great actors couldn't save this terrible script.",
        "Completely ruined what could have been a promising premise. Terrible execution.",
        "Unoriginal, derivative, boring, and insulting to the audience's intelligence.",
        "Failed on every level. Defective upon arrival and zero warranty support. Do not buy.",
        "Boring, tedious, and filled with ridiculous clichés from start to finish. Hated it.",
        "The worst sequel ever created. It completely destroyed the legacy of the original.",
        "Extremely noisy, erratic, and utterly devoid of genuine emotion. Terrible watch.",
        "Zero stars if possible. Completely non-functional, broke immediately, and totally useless.",
        "Pathetic writing and unconvincing visual effects. A total disaster and waste of time.",
        "Terrible battery performance, constantly overheats, and stopped charging after a week.",
        "I walked out of the theater halfway through. Could not bear this awful movie any longer.",
        "Overpriced, underperforming, and arrived badly damaged. Extremely disappointed.",
        "Painful to watch. Horrific dialogue delivered without an ounce of emotion or talent.",
        "A total disaster from beginning to end. Don't even bother watching this awful trash.",
        "Regrettable purchase. It does not match the advertised specifications and feels cheap.",
        "Shallow, pointless, and devoid of any artistic merit or entertainment. Horrible film.",
        "Customer service kept me on hold for an hour before disconnecting. Worst support ever.",
        "The climax made absolutely no sense and left everyone annoyed. Truly terrible ending.",
        "Cheap knockoff that feels brittle, cheap, and broke on the very first day.",
        "Very poor quality. Stopped working after three days and seller refused to help.",
        "Horrible experience! The software is full of bugs and glitches. Avoid this junk.",
        "Worst movie I have seen all year. Utterly boring, pointless, and a waste of money.",
        "Terrible customer service. Nobody replies to emails and phone calls get dropped.",
        "Disastrous purchase. The item arrived broken and looks nothing like the photo.",
        "Awful sound quality, cheap plastic construction, and uncomfortable to wear.",
        "Boring from start to finish. I fell asleep thirty minutes into the film.",
        "Total waste of money! Cheaply made and stopped working almost immediately.",
        "Horribly slow delivery and the product was defective. Will never order again.",
        "The acting was atrocious and the script was embarrassingly bad.",
        "Extremely frustrating experience. Buggy, slow, unresponsive, and keeps crashing.",
        "Do not buy this product! It is a complete scam and does not work at all.",
        "Terrible script, cringe dialogue, and awful visual effects. Avoid this garbage.",
        "Worst customer support ever. Rude representatives and refused to honor the warranty.",
        "Poor performance, sluggish speed, and overheats rapidly. Very bad purchase.",
        "Completely useless item. Broke into pieces after normal use. Total rubbish.",
        "I regret spending money on this. Low quality, bad materials, and looks ugly.",
        "Awful film. Predictable twists, bad acting, and endless boring scenes.",
        "Defective out of the box. Does not turn on and support was utterly unhelpful.",
        "The worst purchase I made this year. Low quality and very disappointing.",
        "Horrible user interface, impossible to navigate, and riddled with glitches.",
        "Painfully bad acting and completely lifeless directing. A disaster.",
        "Fell apart within hours. Cheaply manufactured and extremely shoddy.",
        "Terrible experience from start to finish. Would give negative stars if I could.",
        "Disgusting quality, smelled bad out of the box, and broke immediately."
    ]

    data = [(text, "positive") for text in positives] + [(text, "negative") for text in negatives]
    return pd.DataFrame(data, columns=["text", "sentiment"])


def load_dataset(filepath: str | Path | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Load dataset from a file or generate the seed dataset if file doesn't exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = Path(filepath) if filepath else DATA_RAW_DIR / "sentiment_data.csv"
    
    if target_path.exists() and not force_refresh:
        df = pd.read_csv(target_path)
        if "review" in df.columns and "text" not in df.columns:
            df.rename(columns={"review": "text"}, inplace=True)
    else:
        df = get_seed_dataset()
        df.to_csv(target_path, index=False)
        
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[list[str], list[str], list[str], list[str]]:
    """Clean text and split into train and test sets with stratification."""
    cleaned_texts = [clean_text(t) for t in df["text"]]
    labels = df["sentiment"].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test
