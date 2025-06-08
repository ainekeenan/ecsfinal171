from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

#Import Models

models={
    'Linear': joblib.load('models/linear_model.pkl'),
    'Lasso': joblib.load('models/lasso_model.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'Gradient Boost': joblib.load('models/gradient_boost.pkl')
}

scalers={
    'Linear': joblib.load('models/scaler_linear.pkl'),
    'Lasso': joblib.load('models/scaler_lasso.pkl')
}

feature_columns_linear = joblib.load('models/feature_columns_linear.pkl')
feature_columns_lasso = joblib.load('models/feature_columns_lasso.pkl')
feature_columns_rf = joblib.load('models/feature_columns_rf.pkl')
feature_columns_gb = joblib.load('models/feature_columns_gb.pkl')

AVAILABLE_GENRES = [
    'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 
    'bluegrass', 'blues', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 
    'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 
    'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 
    'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 
    'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 
    'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 
    'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 
    'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 
    'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb', 'new-age', 
    'opera', 'pagode', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 
    'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'reggae', 'reggaeton', 'rock', 'rock-n-roll', 
    'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 
    'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'study', 
    'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'world-music'
]

@app.route("/")
def index():
    return render_template("index.html", genres=AVAILABLE_GENRES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        danceability = float(request.form['danceability'])
        if not(0.0 <= danceability <= 1.0):
            return "Error: Danceability must be between 0.0 and 1.0"
        
        energy = float(request.form['energy'])
        if not(0.0 <= energy <= 1.0):
            return "Error: Energy must be between 0.0 and 1.0"
        
        loudness = float(request.form['loudness'])
        if not(-60.0 <= loudness <= 0.0):
            return "Error: Loudness must be between -60.0 and 0.0"
        
        speechiness = float(request.form['speechiness'])
        if not (0.0 <= speechiness <= 1.0):
            return "Error: Speechiness must be between 0.0 and 1.0"
        
        acousticness = float(request.form['acousticness'])
        if not (0.0 <= acousticness <= 1.0):
            return "Error: Acousticness must be between 0.0 and 1.0"
        
        instrumentalness = float(request.form['instrumentalness'])
        if not (0.0 <= instrumentalness <= 1.0):
            return "Error: Instrumentalness must be between 0.0 and 1.0"

        valence = float(request.form['valence'])
        if not (0.0 <= valence <= 1.0):
            return "Error: Valence must be between 0.0 and 1.0"
        
        tempo = float(request.form['tempo'])
        if not (50.0 <= tempo <= 200.0):
            return "Error: Tempo must be between 50.0 and 200.0"
        
        duration_seconds = float(request.form['duration'])
        if not (30.0 <= duration_seconds <= 600.0):
            return "Error: Duration must be between 30.0 and 600.0"
        duration_ms = duration_seconds * 1000
        
        selected_genre = request.form.get('genre', 'pop')
        if selected_genre not in AVAILABLE_GENRES:
            return f"Error: Invalid genre. Must be one of {', '.join(AVAILABLE_GENRES)}"
        
        model_choice = request.form['model_choice']
        model = models[model_choice]

        input_dict = {feature: 0 for feature in feature_columns_gb}  
        input_dict.update({
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'valence': valence,
            'tempo': tempo,
            'duration_ms': duration_ms,
        })

        # Incoorporate the selected genre
        input_dict[selected_genre] = 1

        input_data = pd.DataFrame([input_dict])

        # Set up data to send to the model 
        if model_choice == 'Linear':
            input_data = input_data[feature_columns_linear]
            input_data = scalers['Linear'].transform(input_data)

        elif model_choice == 'Lasso':
            input_data = input_data[feature_columns_lasso]
            input_data = scalers['Lasso'].transform(input_data)

        elif model_choice == 'Random Forest':
            input_data = input_data[feature_columns_rf]

        else:  
            input_data = input_data[feature_columns_gb]

        prediction = model.predict(input_data)[0]

        if model_choice == 'Gradient Boost':
            prediction = prediction * 100

        prediction = np.clip(prediction, 0, 100)
        prediction = round(prediction, 2)

        return render_template("result.html", prediction=prediction, model_used=model_choice, genre=selected_genre)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)






