import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Function to check keywords in the description
def check_keywords(text):
    num = 100
    try:
        positive_keywords = ["חדש", "מובילאיי", "חיישן", "מולטימדיה", "מטופל", "ללא תאונות", "מקוריים", "מקורי", "טוב", "חסכוני", "פרטית", "מורשה", "שמור"]
        negative_keywords = ["סריטות", "שריטות", "ליסינג", "מונית", "כתמי שמש", "שריטה", "מכה", "מכות"]

        for keyword in positive_keywords:
            if keyword in text:
                num += 20

        for keyword in negative_keywords:
            if keyword in text:
                num -= 20

        return num
    except Exception as e:
        return num

app = Flask(__name__)
rf_model = pickle.load(open("trained_model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html', form_data={}, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    form_data = request.form.to_dict()
    
    # Convert form data to list of features
    features = [
        form_data.get('manufactor', ''),
        form_data.get('Year', ''),
        form_data.get('model', ''),
        form_data.get('Hand', ''),
        form_data.get('Gear', ''),
        form_data.get('capacity_Engine', ''),
        form_data.get('Engine_type', ''),
        form_data.get('Prev_ownership', ''),
        form_data.get('Curr_ownership', ''),
        form_data.get('Pic_num', ''),
        form_data.get('Description', ''),
        form_data.get('Color', ''),
        form_data.get('is_reposted', '')
    ]
    
    final_features_df = pd.DataFrame([features], columns=[
        'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 
        'Prev_ownership', 'Curr_ownership', 'Pic_num', 'Description', 'Color', 'is_reposted'
    ])
    final_features_df['Description'] = final_features_df['Description'].apply(check_keywords)

    prediction = rf_model.predict(final_features_df)[0]

    return render_template('index.html', form_data=form_data, prediction_text='Car predicted price is: {} ₪'.format(prediction))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
