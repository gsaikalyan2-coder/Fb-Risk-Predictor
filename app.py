import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Load dataset ---
data = pd.read_csv("player_injuries_impact.csv")

# Safe date parsing
data['Date of Injury'] = pd.to_datetime(data['Date of Injury'], errors='coerce')
data['Date of return'] = pd.to_datetime(data['Date of return'], errors='coerce')

# Clean player rating columns (before and after injury)
player_rating_cols = [
    'Match1_before_injury_Player_rating',
    'Match2_before_injury_Player_rating',
    'Match3_before_injury_Player_rating',
    'Match1_after_injury_Player_rating',
    'Match2_after_injury_Player_rating',
    'Match3_after_injury_Player_rating'
]

for col in player_rating_cols:
    data[col] = data[col].astype(str).str.replace('(S)', '', regex=False)
    # Ensure any string that cannot be parsed as a number (like '7.46.15.8') is converted to NaN
    data[col] = pd.to_numeric(data[col].replace('N.A.', pd.NA), errors='coerce')
    data[col] = data[col].fillna(0).astype(float) # Ensure final type is float

# --- Ruthless Risk Labeling ---
def classify_risk(row):
    risk = "Low"
    if pd.notnull(row['Injury']):
        if pd.notnull(row['Date of Injury']) and pd.notnull(row['Date of return']):
            recovery_days = (row['Date of return'] - row['Date of Injury']).days
            if recovery_days > 60 or row['FIFA rating'] < 70 or row['Age'] > 32:
                risk = "High"
            elif recovery_days > 30 or row['FIFA rating'] < 75 or row['Age'] > 28:
                risk = "Medium"
            else:
                risk = "Low"
        else:
            risk = "Medium"

    # Performance-based rules
    # These columns should now be float due to the cleaning loop
    before_avg = row[['Match1_before_injury_Player_rating',
                      'Match2_before_injury_Player_rating',
                      'Match3_before_injury_Player_rating']].mean(skipna=True)
    after_avg = row[['Match1_after_injury_Player_rating',
                     'Match2_after_injury_Player_rating',
                     'Match3_after_injury_Player_rating']].mean(skipna=True)

    if pd.notnull(before_avg) and pd.notnull(after_avg):
        if after_avg < before_avg - 1.0:
            risk = "High"
        elif after_avg < before_avg:
            risk = "Medium"

    return risk

data['RiskLevel'] = data.apply(classify_risk, axis=1)

# --- Balance dataset ---
high = data[data['RiskLevel']=="High"]
medium = data[data['RiskLevel']=="Medium"]
low = data[data['RiskLevel']=="Low"]

max_size = max(len(high), len(medium), len(low))
high_resampled = resample(high, replace=True, n_samples=max_size, random_state=42)
medium_resampled = resample(medium, replace=True, n_samples=max_size, random_state=42)
low_resampled = resample(low, replace=True, n_samples=max_size, random_state=42)
data_balanced = pd.concat([high_resampled, medium_resampled, low_resampled])

# --- Features & Labels ---
X = data_balanced[['Age','FIFA rating',
                   'Match1_before_injury_Player_rating',
                   'Match2_before_injury_Player_rating',
                   'Match3_before_injury_Player_rating',
                   'Match1_after_injury_Player_rating',
                   'Match2_after_injury_Player_rating',
                   'Match3_after_injury_Player_rating']].fillna(0).values
y = data_balanced['RiskLevel'].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# --- Train model ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# --- Single Player Prediction ---
classes = ["Low Risk", "Medium Risk", "High Risk"]
single_history = {}
compare_history = []

def predict_with_history(player_name):
    if player_name not in data['Name'].values:
        return "Invalid player name.", None, None, "\n".join(single_history.keys())
    player = data[data['Name'] == player_name].iloc[0]
    structured = np.array([[player['Age'], player['FIFA rating'],
                            player['Match1_before_injury_Player_rating'],
                            player['Match2_before_injury_Player_rating'],
                            player['Match3_before_injury_Player_rating'],
                            player['Match1_after_injury_Player_rating'],
                            player['Match2_after_injury_Player_rating'],
                            player['Match3_after_injury_Player_rating']]])
    structured = scaler.transform(structured)
    prediction = model.predict(structured)[0]
    risk_class = classes[np.argmax(prediction)]
    color_map = {"Low Risk":"🟢 Low Risk", "Medium Risk":"🟠 Medium Risk", "High Risk":"🔴 High Risk"}
    info = f"""
    **Name:** {player['Name']}
    **Team:** {player['Team Name']}
    **Position:** {player['Position']}
    **Age:** {player['Age']}
    **FIFA Rating:** {player['FIFA rating']}
    """
    single_history[player_name] = color_map[risk_class]
    return color_map[risk_class], info, {classes[i]: float(prediction[i]) for i in range(len(classes))}, "\n".join([f"{k}: {v}" for k,v in single_history.items()])

# --- Compare Two Players ---
def compare_players(player1, player2):
    results, probs = [], {}
    for p in [player1, player2]:
        if p not in data['Name'].values:
            results.append(f"{p}: Invalid name")
            probs[p] = None
        else:
            structured = np.array([[data.loc[data['Name']==p,'Age'].values[0],
                                    data.loc[data['Name']==p,'FIFA rating'].values[0],
                                    data.loc[data['Name']==p,'Match1_before_injury_Player_rating'].values[0],
                                    data.loc[data['Name']==p,'Match2_before_injury_Player_rating'].values[0],
                                    data.loc[data['Name']==p,'Match3_before_injury_Player_rating'].values[0],
                                    data.loc[data['Name']==p,'Match1_after_injury_Player_rating'].values[0],
                                    data.loc[data['Name']==p,'Match2_after_injury_Player_rating'].values[0],
                                    data.loc[data['Name']==p,'Match3_after_injury_Player_rating'].values[0]]])
            structured = scaler.transform(structured)
            prediction = model.predict(structured)[0]
            risk_class = classes[np.argmax(prediction)]
            results.append(f"{p}: {risk_class}")
            probs[p] = prediction
    compare_history.append("\n".join(results))
    if len(compare_history) > 5:
        compare_history.pop(0)
    fig, ax = plt.subplots()
    bar_width = 0.35
    indices = np.arange(len(classes))
    if probs[player1] is not None and probs[player2] is not None:
        ax.bar(indices, probs[player1], bar_width, label=player1)
        ax.bar(indices + bar_width, probs[player2], bar_width, label=player2)
        ax.set_xticks(indices + bar_width/2)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylabel("Probability")
        ax.set_title("Risk Probability Comparison")
    prob_text = ""
    for p in [player1, player2]:
        if probs[p] is not None:
            prob_text += f"\n{p} Probabilities:\n"
            for i, c in enumerate(classes):
                prob_text += f"  {c}: {probs[p][i]*100:.1f}%\n"
    return "\n".join(results), "\n".join(compare_history), fig, prob_text

# --- Safer Graph Functions ---
def risk_distribution_graph():
    if 'RiskLevel' not in data.columns or data['RiskLevel'].dropna().empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No RiskLevel data",ha='center'); return fig
    counts = data['RiskLevel'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['green','orange','red'])
    ax.set_title("Overall Risk Distribution")
    return fig

def age_vs_risk_graph():
    if 'Age' not in data.columns or data['Age'].dropna().empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No Age data",ha='center'); return fig
    fig, ax = plt.subplots()
    data.boxplot(column='Age', by='RiskLevel', ax=ax, grid=False)
    ax.set_title("Age vs Risk Level"); ax.set_ylabel("Age")
    return fig
def fifa_vs_risk_graph():
    if 'FIFA rating' not in data.columns or data['FIFA rating'].dropna().empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No FIFA rating data",ha='center'); return fig
    avg_ratings = data.groupby('RiskLevel')['FIFA rating'].mean()
    fig, ax = plt.subplots()
    avg_ratings.plot(kind='bar', color=['green','orange','red'], ax=ax)
    ax.set_title("Average FIFA Rating by Risk Level")
    ax.set_ylabel("FIFA Rating")
    return fig

def team_risk_graph():
    if 'Team Name' not in data.columns or data['Team Name'].dropna().empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No Team data",ha='center'); return fig
    team_counts = data.groupby(['Team Name','RiskLevel']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10,6))
    team_counts.plot(kind='bar', stacked=True, ax=ax, color=['green','orange','red'])
    ax.set_title("Risk Distribution by Team")
    ax.set_ylabel("Number of Players")
    return fig

def performance_drop_graph():
    required_cols = ['Match1_before_injury_Player_rating','Match1_after_injury_Player_rating']
    if not set(required_cols).issubset(data.columns):
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No performance rating data",ha='center'); return fig
    before = data[['Match1_before_injury_Player_rating',
                   'Match2_before_injury_Player_rating',
                   'Match3_before_injury_Player_rating']].mean(axis=1)
    after = data[['Match1_after_injury_Player_rating',
                  'Match2_after_injury_Player_rating',
                  'Match3_after_injury_Player_rating']].mean(axis=1)
    fig, ax = plt.subplots()
    ax.scatter(before, after, c=data['RiskLevel'].map({'Low':'green','Medium':'orange','High':'red'}))
    ax.set_xlabel("Average Rating Before Injury")
    ax.set_ylabel("Average Rating After Injury")
    ax.set_title("Performance Drop Before vs After Injury")
    return fig

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## ⚽ Football Injury Risk Predictor")

    with gr.Tab("Single Player Prediction"):
        player_dropdown = gr.Dropdown(choices=list(data['Name'].unique()), label="Select Footballer")
        output_text = gr.Textbox(label="Risk Level")
        player_info = gr.Markdown(label="Player Info")
        prob_output = gr.Label(label="Risk Probabilities")
        history_output = gr.Textbox(label="Prediction History")
        player_dropdown.change(predict_with_history, inputs=player_dropdown,
                               outputs=[output_text, player_info, prob_output, history_output])

    with gr.Tab("Compare Two Players"):
        player1 = gr.Dropdown(choices=list(data['Name'].unique()), label="Player 1")
        player2 = gr.Dropdown(choices=list(data['Name'].unique()), label="Player 2")
        compare_output = gr.Textbox(label="Comparison Result")
        compare_history_output = gr.Textbox(label="Comparison History")
        prob_graph = gr.Plot(label="Risk Probability Graph")
        prob_text = gr.Textbox(label="Risk Probability Breakdown")
        compare_btn = gr.Button("Compare")
        compare_btn.click(compare_players, inputs=[player1, player2],
                          outputs=[compare_output, compare_history_output, prob_graph, prob_text])

    with gr.Tab("Graphs"):
        dist_btn = gr.Button("Risk Distribution")
        dist_plot = gr.Plot()
        dist_btn.click(risk_distribution_graph, inputs=None, outputs=dist_plot)

        age_btn = gr.Button("Age vs Risk")
        age_plot = gr.Plot()
        age_btn.click(age_vs_risk_graph, inputs=None, outputs=age_plot)

        fifa_btn = gr.Button("FIFA Rating vs Risk")
        fifa_plot = gr.Plot()
        fifa_btn.click(fifa_vs_risk_graph, inputs=None, outputs=fifa_plot)

        team_btn = gr.Button("Team Risk Distribution")
        team_plot = gr.Plot()
        team_btn.click(team_risk_graph, inputs=None, outputs=team_plot)

        perf_btn = gr.Button("Performance Drop")
        perf_plot = gr.Plot()
        perf_btn.click(performance_drop_graph, inputs=None, outputs=perf_plot)

demo.launch(share=True)
