import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Simulate data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "TierID": np.random.choice(["M001", "M002", "M003"], n),
    "Time": np.random.choice([0, 24, 48], n),
    "Dosis": np.random.choice([1.0, 1.5, 2.0], n),
    "Gewebe": np.random.choice(["Leber", "Niere"], n),
    "Jod": np.random.uniform(0.05, 0.8, n),
})
df["Zscore"] = (df["Jod"] - df["Jod"].mean()) / df["Jod"].std()

# Create 3D scatter plot
fig = go.Figure(data=go.Scatter3d(
    x=df['Time'],
    y=df['Dosis'],
    z=df['Jod'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['Zscore'],  # or df['Jod']
        colorscale='Viridis',
        colorbar=dict(title="Z-Score")
    ),
    text=df.apply(lambda row: f"TierID: {row['TierID']}<br>Gewebe: {row['Gewebe']}", axis=1),
    hoverinfo='text'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Time (h)',
        yaxis_title='Dose',
        zaxis_title='Jod'
    ),
    title="3D Jodverteilung nach Zeit, Dosis, Gewebe",
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()
