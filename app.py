import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from antropy import perm_entropy, spectral_entropy, svd_entropy, sample_entropy, petrosian_fd, katz_fd, higuchi_fd

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

# Load pretrained model
MODEL_PATH = 'clf.pkl'
model = joblib.load(MODEL_PATH)

# Load Subjects.csv for native language lookup
SUBJECTS_CSV_PATH = 'Subjects.csv'
subjects_df = pd.read_csv(SUBJECTS_CSV_PATH)
subjects_df['Subject ID'] = subjects_df['Subject ID'].str.strip().str.lower()
subjects_df['Mother Language'] = subjects_df['Mother Language'].str.strip().str.lower()

# Load full features dataset to get feature columns and fit scaler
FEATURES_PKL_PATH = 'EEG_Features_Full.pkl'
full_features_df = pd.read_pickle(FEATURES_PKL_PATH)

# Drop metadata columns from features and fit scaler
feature_cols = full_features_df.columns.drop(['subject_id', 'experiment_id', 'file'], errors='ignore')
scaler = StandardScaler()
scaler.fit(full_features_df[feature_cols])

# --- Feature Extraction Functions ---

def extract_statistical_features(df):
    features = {}
    for ch in df.columns:
        signal = df[ch].values
        features[f'{ch}_mean'] = np.mean(signal)
        features[f'{ch}_std'] = np.std(signal)
        features[f'{ch}_var'] = np.var(signal)
        features[f'{ch}_range'] = np.max(signal) - np.min(signal)
        features[f'{ch}_kurtosis'] = kurtosis(signal)
        features[f'{ch}_skewness'] = skew(signal)
    return features

def extract_psd_features(df):
    features = {}
    fs = 256  # sampling frequency
    for ch in df.columns:
        signal = df[ch].values
        freqs, psd = welch(signal, fs=fs)
        features[f'{ch}_psd_mean'] = np.mean(psd)
        features[f'{ch}_psd_std'] = np.std(psd)
    return features

def custom_dfa(signal, order=1):
    x = np.cumsum(signal - np.mean(signal))
    L = np.floor(np.logspace(np.log10(4), np.log10(len(x)//4), num=10)).astype(int)
    F = []
    for l in L:
        if l < 4:
            continue
        shape = (len(x) // l, l)
        if shape[0] < 2:
            continue
        X = np.reshape(x[:shape[0]*l], shape)
        RMS = []
        for segment in X:
            time = np.arange(l)
            coeffs = np.polyfit(time, segment, order)
            trend = np.polyval(coeffs, time)
            RMS.append(np.sqrt(np.mean((segment - trend)**2)))
        F.append(np.mean(RMS))
    log_L = np.log(L[:len(F)])
    log_F = np.log(F)
    alpha = np.polyfit(log_L, log_F, 1)[0]
    return alpha

def extract_entropy_fractal_features(df):
    features = {}
    fs = 256
    for ch in df.columns:
        signal = df[ch].values
        features[f'{ch}_perm_entropy'] = perm_entropy(signal, normalize=True)
        features[f'{ch}_spectral_entropy'] = spectral_entropy(signal, sf=fs, method='welch', normalize=True)
        features[f'{ch}_svd_entropy'] = svd_entropy(signal)
        features[f'{ch}_sample_entropy'] = sample_entropy(signal)
        features[f'{ch}_petrosian_fd'] = petrosian_fd(signal)
        features[f'{ch}_katz_fd'] = katz_fd(signal)
        features[f'{ch}_higuchi_fd'] = higuchi_fd(signal)
        features[f'{ch}_dfa'] = custom_dfa(signal)
    return features

def extract_features_from_df(df):
    # Drop index/time column if it exists
    if 'Unnamed' in df.columns[0] or df.columns[0] == '':
        df = df.iloc[:, 1:]
    stats = extract_statistical_features(df)
    psd = extract_psd_features(df)
    ent = extract_entropy_fractal_features(df)
    combined = {**stats, **psd, **ent}
    return combined

def aggregate_features(list_of_feature_dicts):
    df = pd.DataFrame(list_of_feature_dicts)
    agg = df.mean(axis=0).to_dict()
    return agg

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        subject_id = request.form.get('subject_id', '').strip().lower()
        stimulus_language = request.form.get('stimulus_language', '').strip().lower()
        files = request.files.getlist('eeg_files')

        if not subject_id or not stimulus_language:
            flash("Please provide Subject ID and Stimulus Language.")
            return redirect(request.url)
        if not files or files[0].filename == '':
            flash("Please upload at least one EEG CSV file.")
            return redirect(request.url)

        feature_dicts = []
        for f in files:
            try:
                df = pd.read_csv(io.StringIO(f.stream.read().decode("UTF8")))
            except Exception as e:
                flash(f"Error reading file {f.filename}: {e}")
                return redirect(request.url)
            try:
                features = extract_features_from_df(df)
                feature_dicts.append(features)
            except Exception as e:
                flash(f"Error extracting features from {f.filename}: {e}")
                return redirect(request.url)

        agg_features = aggregate_features(feature_dicts)
        input_df = pd.DataFrame([agg_features])

        missing_cols = set(feature_cols) - set(input_df.columns)
        if missing_cols:
            flash(f"Missing features for prediction: {missing_cols}")
            return redirect(request.url)

        input_df = input_df[feature_cols]
        scaled_features = scaler.transform(input_df)

        # Predict and apply modulo 2 logic
        prediction = model.predict(scaled_features)[0]
        prediction_mod = prediction % 2

        if prediction_mod == 0:
            predicted_label = 'non-native'
        else:
            predicted_label = 'native'

        native_language_row = subjects_df[subjects_df['Subject ID'] == subject_id]
        if native_language_row.empty:
            native_language = "No data is given for crossreferencing WRT this subect ID"
        else:
            native_language = native_language_row.iloc[0]['Mother Language']

        expected_status = 'native' if stimulus_language == native_language else 'non-native'

        result = {
            'subject_id': subject_id.upper(),
            'stimulus_language': stimulus_language.capitalize(),
            'native_language': native_language.capitalize(),
            'expected_status': expected_status.capitalize(),
            'predicted_status': predicted_label.capitalize(),
            'correct_prediction': predicted_label.lower() == expected_status.lower()
        }

        return render_template('result.html', result=result)

    return render_template('index.html')

# New AJAX route to get native language for a subject ID
@app.route('/get_native_language', methods=['POST'])
def get_native_language():
    data = request.get_json()
    subject_id = data.get('subject_id', '').strip().lower()
    if not subject_id:
        return jsonify({'native_language': '', 'message': 'Please enter a Subject ID.'})

    native_language_row = subjects_df[subjects_df['Subject ID'] == subject_id]
    if native_language_row.empty:
        return jsonify({'native_language': '', 'message': 'Subject data is not available in the subject database.'})
    else:
        native_language = native_language_row.iloc[0]['Mother Language']
        return jsonify({'native_language': native_language.capitalize(), 'message': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
