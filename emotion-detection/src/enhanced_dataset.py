"""
Enhanced Dataset Generator for Better Emotion Detection Performance
Creates a larger, more balanced dataset for improved model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

def create_enhanced_emotion_dataset():
    """
    Create a comprehensive emotion dataset with more examples and better balance
    This simulates a real-world dataset like those from Kaggle
    """
    
    # Enhanced Joy examples (positive emotions)
    joy_texts = [
        "I'm absolutely thrilled about this amazing opportunity!",
        "Feeling fantastic today! Everything is going perfectly!",
        "So excited about the weekend plans with friends!",
        "Just got promoted at work! Best day ever!",
        "My baby took her first steps today! Pure joy!",
        "Won the lottery! Can't believe this is happening!",
        "Amazing vacation in Hawaii! Living the dream!",
        "Just married the love of my life! Ecstatic!",
        "Graduated with honors! All the hard work paid off!",
        "Surprise party was incredible! Feeling blessed!",
        "This is the happiest day of my life!",
        "I love spending time with my family during holidays!",
        "Got accepted to my dream university! Over the moon!",
        "The concert last night was absolutely phenomenal!",
        "Feeling grateful for all the wonderful people in my life!",
        "Just achieved my biggest goal! I'm on cloud nine!",
        "The weather is beautiful and I feel amazing!",
        "My favorite team won the championship! Celebrating!",
        "Just received the best news ever! Dancing with joy!",
        "This delicious meal is making me so happy!",
        "I'm beaming with pride after today's presentation!",
        "Life is wonderful when you're surrounded by love!",
        "Feeling optimistic about the future ahead!",
        "Just finished an incredible book! Loved every page!",
        "The sunrise this morning was breathtakingly beautiful!",
        "My dog's happy face always makes me smile!",
        "Accomplished something I never thought possible!",
        "Feeling energized and ready to take on the world!",
        "The kindness of strangers today restored my faith!",
        "Just had the most amazing conversation with a friend!"
    ]
    
    # Enhanced Anger examples (frustration, rage, irritation)
    anger_texts = [
        "This situation is making me extremely frustrated!",
        "Can't believe how unfair this treatment is!",
        "Traffic jam is making me furious! Running late!",
        "Terrible customer service! Completely unacceptable!",
        "My computer crashed and I lost all my work!",
        "Neighbors are being incredibly inconsiderate!",
        "Got charged twice for the same purchase! Outrageous!",
        "Boss rejected my proposal without even reading it!",
        "Parking ticket for no good reason! This is ridiculous!",
        "Internet has been down for hours! So frustrating!",
        "I'm absolutely livid about this injustice!",
        "This incompetence is driving me up the wall!",
        "How dare they treat me with such disrespect!",
        "I'm fed up with these constant interruptions!",
        "This bureaucratic nonsense is infuriating!",
        "The referee made such a terrible call! I'm furious!",
        "My flight got cancelled for the third time! Enraged!",
        "This malfunctioning equipment is making me mad!",
        "I can't stand people who don't keep their promises!",
        "The noise from construction is driving me crazy!",
        "This political corruption makes my blood boil!",
        "I'm sick and tired of being taken advantage of!",
        "The rude behavior of that person was appalling!",
        "This software keeps crashing! It's so aggravating!",
        "I'm outraged by the poor quality of this service!",
        "The long wait time is testing my patience!",
        "This unfair decision has me seeing red!",
        "I'm irritated by the constant spam emails!",
        "The arrogance of some people is just maddening!",
        "This broken promise has left me feeling betrayed!"
    ]
    
    # Enhanced Fear examples (anxiety, worry, terror)
    fear_texts = [
        "Really scared about the upcoming surgery next week.",
        "Terrified of flying but have to take this business trip.",
        "Worried sick about my job interview tomorrow.",
        "The storm outside is making me anxious and fearful.",
        "Afraid I won't pass the final exam despite studying.",
        "Nervous about moving to a new city all alone.",
        "Scared of what the test results might reveal.",
        "Anxious about giving a presentation to the board.",
        "Frightened by the strange noises in the house.",
        "Panic attack thinking about the deadline approaching.",
        "I'm terrified of what might happen next.",
        "The thought of public speaking fills me with dread.",
        "Worried about my family's safety during the storm.",
        "Afraid of losing my job in these uncertain times.",
        "The horror movie last night gave me nightmares.",
        "Anxious about the medical procedure scheduled tomorrow.",
        "Scared of walking alone in this neighborhood at night.",
        "Worried about my child's first day at school.",
        "The turbulence during the flight was terrifying!",
        "Afraid of making the wrong decision about my career.",
        "Nervous about meeting my partner's parents.",
        "The dark alley ahead is making me feel uneasy.",
        "Worried about the impact of climate change.",
        "Scared of heights but have to climb this ladder.",
        "Anxious about the upcoming financial audit.",
        "Afraid of confronting my boss about the issue.",
        "The creepy atmosphere in this place is unsettling.",
        "Terrified of losing someone I love.",
        "Worried about my ability to handle this responsibility.",
        "The uncertainty of the future is keeping me awake."
    ]
    
    # Enhanced Sadness examples (depression, melancholy, grief)
    sadness_texts = [
        "Feeling really down and depressed today.",
        "Heartbroken about losing my beloved pet.",
        "So sad to see my best friend moving away.",
        "Mourning the loss of my grandmother last week.",
        "Disappointed that my vacation got cancelled.",
        "Feeling lonely since my roommate moved out.",
        "Devastated by the news about the company closing.",
        "Melancholy mood because of the rainy weather.",
        "Sad to see how divided our community has become.",
        "Gloomy atmosphere after hearing the bad news.",
        "I'm overwhelmed with grief after the funeral.",
        "Feeling empty inside since the breakup.",
        "The documentary about poverty left me heartbroken.",
        "Sad memories keep flooding back today.",
        "Feeling blue about missing the family gathering.",
        "The ending of that movie made me cry.",
        "Depressed about not achieving my goals this year.",
        "Feeling sorrowful about the state of the world.",
        "The grey skies match my melancholic mood.",
        "Heartache from saying goodbye to old friends.",
        "Feeling dejected after the rejection letter.",
        "The abandoned puppy made me incredibly sad.",
        "Grieving the loss of my childhood home.",
        "Feeling down about my recent failures.",
        "The tragic news story brought tears to my eyes.",
        "Sad about the end of summer vacation.",
        "Feeling despondent about the future.",
        "The lonely old man in the park made me weep.",
        "Mourning the end of a beautiful relationship.",
        "Feeling sorrowful about missed opportunities."
    ]
    
    # Enhanced Neutral examples (factual, observational, mundane)
    neutral_texts = [
        "Just another ordinary day at the office.",
        "Weather is okay today, not too hot or cold.",
        "Had lunch, now back to work on the project.",
        "Commute was normal, no delays or issues.",
        "Finished reading a book, it was alright.",
        "Went to the store, bought some groceries.",
        "Attended the meeting, discussed quarterly results.",
        "Watched a documentary about history last night.",
        "Completed the daily tasks as scheduled.",
        "Regular workout session at the gym today.",
        "The train arrived on time as usual.",
        "Checked my email and responded to messages.",
        "The meeting is scheduled for 3 PM tomorrow.",
        "Had a sandwich for lunch, nothing special.",
        "The temperature outside is 72 degrees.",
        "Finished the report and submitted it on time.",
        "Walked to the park, sat on a bench for a while.",
        "The grocery store was moderately busy today.",
        "Updated my calendar with next week's appointments.",
        "Filled up the gas tank before heading home.",
        "The library was quiet, perfect for studying.",
        "Organized my desk and filed important documents.",
        "The coffee shop had their usual selection available.",
        "Took the elevator to the fifth floor.",
        "The parking lot was half full this morning.",
        "Reviewed the contract and made some notes.",
        "The restaurant had a standard menu selection.",
        "Picked up dry cleaning on the way home.",
        "The bank was open during regular business hours.",
        "Sorted through mail and paid some bills."
    ]
    
    # Create balanced dataset
    emotions_data = {
        'text': joy_texts + anger_texts + fear_texts + sadness_texts + neutral_texts,
        'emotion': (['joy'] * len(joy_texts) + 
                   ['anger'] * len(anger_texts) + 
                   ['fear'] * len(fear_texts) + 
                   ['sadness'] * len(sadness_texts) + 
                   ['neutral'] * len(neutral_texts))
    }
    
    df = pd.DataFrame(emotions_data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Enhanced dataset created with {len(df)} samples:")
    print(df['emotion'].value_counts())
    
    return df

def create_mega_dataset():
    """
    Create an even larger dataset by adding variations and synthetic examples
    """
    base_df = create_enhanced_emotion_dataset()
    
    # Add variations using different phrasings
    variations = []
    
    # Simple text variations for more training data
    variation_patterns = {
        'joy': [
            "I feel {} about this situation",
            "This makes me feel {}",
            "I'm {} right now",
            "Feeling {} today",
            "{} is what I'm experiencing"
        ],
        'anger': [
            "I'm {} about this issue",
            "This situation makes me {}",
            "Feeling {} right now",
            "I'm {} with this behavior",
            "This {} me so much"
        ],
        'fear': [
            "I'm {} about what might happen",
            "Feeling {} about this situation",
            "This makes me {}",
            "I'm {} of the outcome",
            "Experiencing {} right now"
        ],
        'sadness': [
            "I feel {} about this news",
            "This makes me {}",
            "Feeling {} today",
            "I'm {} about the situation",
            "This brings me {}"
        ],
        'neutral': [
            "This is {} normal",
            "Just a {} day",
            "Nothing {} happening",
            "Everything seems {}",
            "A {} ordinary situation"
        ]
    }
    
    emotion_words = {
        'joy': ['happy', 'excited', 'thrilled', 'delighted', 'cheerful', 'elated'],
        'anger': ['angry', 'furious', 'irritated', 'frustrated', 'mad', 'annoyed'],
        'fear': ['scared', 'worried', 'anxious', 'nervous', 'afraid', 'terrified'],
        'sadness': ['sad', 'depressed', 'down', 'melancholy', 'heartbroken', 'sorrowful'],
        'neutral': ['pretty', 'regular', 'particularly', 'quite', 'fairly', 'rather']
    }
    
    # Generate variations
    for emotion in emotion_words:
        patterns = variation_patterns[emotion]
        words = emotion_words[emotion]
        
        for pattern in patterns[:3]:  # Use first 3 patterns
            for word in words[:4]:  # Use first 4 words
                if emotion == 'neutral':
                    text = pattern.format(word)
                else:
                    text = pattern.format(word)
                
                variations.append({
                    'text': text,
                    'emotion': emotion
                })
    
    variation_df = pd.DataFrame(variations)
    
    # Combine original and variations
    mega_df = pd.concat([base_df, variation_df], ignore_index=True)
    mega_df = mega_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nMega dataset created with {len(mega_df)} samples:")
    print(mega_df['emotion'].value_counts())
    
    return mega_df

def save_dataset(df, filename='enhanced_emotion_dataset.csv'):
    """
    Save the dataset to a CSV file
    """
    filepath = f'../data/{filename}'
    df.to_csv(filepath, index=False)
    print(f"\nDataset saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    print("Creating Enhanced Emotion Dataset...")
    
    # Create enhanced dataset
    enhanced_df = create_enhanced_emotion_dataset()
    
    # Create mega dataset
    print("\nCreating Mega Dataset with variations...")
    mega_df = create_mega_dataset()
    
    # Save both datasets
    save_dataset(enhanced_df, 'enhanced_emotion_dataset.csv')
    save_dataset(mega_df, 'mega_emotion_dataset.csv')
    
    print("\n✅ Enhanced datasets created successfully!")
    print("\nDataset Statistics:")
    print(f"Enhanced Dataset: {len(enhanced_df)} samples")
    print(f"Mega Dataset: {len(mega_df)} samples")
    
    print("\nEmotion Distribution (Mega Dataset):")
    for emotion, count in mega_df['emotion'].value_counts().items():
        print(f"  {emotion.title()}: {count} samples")