from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

try:
    classifier = pipeline("sentiment-analysis")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

data = """song_title,artist,mood,region,genre
"walking on sunshine","katrina & the waves","happy","asia","pop"
"stairway to heaven","led zeppelin","happy","asia","rock"
"moonlight sonata","beethoven","happy","asia","classical"
"gangnam style","psy","happy","asia","hip-hop"
"faded","alan walker","happy","asia","electronic"
"fix you","coldplay","happy","europe","pop"
"hotel california","eagles","happy","europe","rock"
"vienna","billy joel","happy","europe","classical"
"uptown funk","mark ronson ft. bruno mars","happy","europe","hip-hop"
"lean on","major lazer","happy","europe","electronic"
"take me home, country roads","john denver","happy","north america","pop"
"bohemian rhapsody","queen","happy","north america","rock"
"clair de lune","debussy","happy","north america","classical"
"empire state of mind","jay-z & alicia keys","happy","north america","hip-hop"
"sunflower","post malone","happy","north america","electronic"
"shape of you","ed sheeran","happy","south america","pop"
"paradise city","guns n' roses","happy","south america","rock"
"la cumparsita","carlos gardel","happy","south america","classical"
"mi gente","j balvin","happy","south america","hip-hop"
"the middle","zedd","happy","south america","electronic"
"halo","beyoncé","happy","africa","pop"
"africa","toto","happy","africa","rock"
"pata pata","miriam makeba","happy","africa","classical"
"this is america","childish gambino","happy","africa","hip-hop"
"on my mind","diplo","happy","africa","electronic"
"perfect","ed sheeran","happy","oceania","pop"
"sweet child o' mine","guns n' roses","happy","oceania","rock"
"waltzing matilda","traditional","happy","oceania","classical"
"thriller","michael jackson","happy","oceania","hip-hop"
"titanium","david guetta","happy","oceania","electronic"
"let it go","idina menzel","sad","asia","pop"
"creep","radiohead","sad","asia","rock"
"ave maria","schubert","sad","asia","classical"
"lose yourself","eminem","sad","asia","hip-hop"
"lights","ellie goulding","sad","asia","electronic"
"wish you were here","pink floyd","sad","europe","pop"
"zombie","cranberries","sad","europe","rock"
"requiem","mozart","sad","europe","classical"
"mockingbird","eminem","sad","europe","hip-hop"
"fade into you","mazzy star","sad","europe","electronic"
"someone like you","adele","sad","north america","pop"
"november rain","guns n' roses","sad","north america","rock"
"adagio for strings","samuel barber","sad","north america","classical"
"god's plan","drake","sad","north america","hip-hop"
"stay","zedd & alessia cara","sad","north america","electronic"
"hello","adele","sad","south america","pop"
"black","pearl jam","sad","south america","rock"
"adios muchachos","carlos gardel","sad","south america","classical"
"feel good inc","gorillaz","sad","south america","hip-hop"
"animals","martin garrix","sad","south america","electronic"
"yesterday","the beatles","sad","africa","pop"
"smells like teen spirit","nirvana","sad","africa","rock"
"nocturne in e-flat","chopin","sad","africa","classical"
"heartless","kanye west","sad","africa","hip-hop"
"ocean eyes","billie eilish","sad","africa","electronic"
"piano man","billy joel","neutral","oceania","pop"
"don't stop believin'","journey","neutral","oceania","rock"
"clair de lune","debussy","neutral","oceania","classical"
"gangsta's paradise","coolio","neutral","oceania","hip-hop"
"the nights","avicii","neutral","oceania","electronic"
"""

df = pd.read_csv(io.StringIO(data))


@app.route('/analyze_sentiment', methods=['POST'])
def analyze():
    if classifier is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text"}), 400

        # Analyze sentiment
        result = classifier(text)[0]
        sentiment_label = result['label']

        # Map sentiment to mood
        sentiment_to_mood = {
            "NEGATIVE": "sad",
            "POSITIVE": "happy",
            "NEUTRAL": "neutral"
        }
        mood = sentiment_to_mood.get(sentiment_label, "Neutral")  # Default to "Neutral" if no match

        # Return both sentiment and mood
        return jsonify({"sentiment": result, "mood": mood})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong"}), 500


@app.route('/get_music', methods=['POST'])
def get_music():
    try:
        data = request.get_json()
        mood = data.get('mood')
        region = data.get('region')
        preference = data.get('preference')

        # Create a copy of the dataset to avoid modifying the original
        filtered_songs = df.copy()

        # Assign relevance scores for each filter
        filtered_songs['relevance'] = 0
        if mood:
            filtered_songs['relevance'] += (filtered_songs['mood'] == mood).astype(int) * 3  # Higher weight for mood
        if region:
            filtered_songs['relevance'] += (filtered_songs['region'] == region).astype(int) * 2  # Medium weight for region
        if preference:
            filtered_songs['relevance'] += (
                (filtered_songs['genre'] == preference).astype(int) +
                (filtered_songs['artist'] == preference).astype(int) * 2
            )  # Preference for artist gets slightly higher weight

        # Filter out songs with zero relevance
        filtered_songs = filtered_songs[filtered_songs['relevance'] > 0]

        # Step 1: Strict match - prioritize highest relevance
        if not filtered_songs.empty:
            filtered_songs = filtered_songs.sort_values(by='relevance', ascending=False)

        # Step 2: Relax filters incrementally
        if filtered_songs.empty:
            if mood or region:
                filtered_songs = df[(df['mood'] == mood) | (df['region'] == region)]
            if filtered_songs.empty and preference:
                filtered_songs = df[(df['genre'] == preference) | (df['artist'] == preference)]

        # Step 3: Fallback to the entire dataset if no match
        if filtered_songs.empty:
            filtered_songs = df.sample(n=10)  # Return a random selection if all filters fail

        # Prepare response
        songs_list = filtered_songs.drop(columns=['relevance']).to_dict(orient='records')
        return jsonify({"songs": songs_list})

    except Exception as e:
        print(f"Error getting music: {e}")
        return jsonify({"error": "Error fetching music"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
