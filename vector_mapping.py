import pandas as pd

# Load your IPA reference table
df = pd.read_csv("Data/ipa-data.csv")

# Select relevant features
features = df[["Symbol", "Voicing", "Place_of_Articulation", "Manner_of_Articulation"]].dropna()

# One-hot encode the categorical features
encoded = pd.get_dummies(features.set_index("Symbol"))

# Convert to dictionary
ipa_feature_dict = encoded.to_dict(orient="index")

# Example: show [p], [b], [m]
for sym in ["p", "b", "m"]:
    print(f"{sym}: {ipa_feature_dict.get(sym)}")
