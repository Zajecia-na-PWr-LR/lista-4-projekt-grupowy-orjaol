import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = 'neo.csv'
df = pd.read_csv(file_path)

print("--- INFO ---")
print(df.info())
print("\n          Brak nam:              ")
print(df.isnull().sum())
# apparently nic god damn

print("\n--- Patrzymy ile obiektow: 'hazardous' ---")
print(df['hazardous'].value_counts(normalize=True) * 100)

df_clean = df.copy()

#id i name idzie do domu z braku wartości dla modelu 
df_clean = df_clean.drop(['id', 'name'], axis=1)

# wszystkie to ziemia wiec byeee
if df_clean['orbiting_body'].nunique() == 1:
    df_clean = df_clean.drop('orbiting_body', axis=1)

if df_clean['sentry_object'].nunique() == 1:
    df_clean = df_clean.drop('sentry_object', axis=1)
elif 'sentry_object' in df_clean.columns:
    df_clean['sentry_object'] = df_clean['sentry_object'].astype(int)

df_clean['hazardous'] = df_clean['hazardous'].astype(int)

X = df_clean.drop('hazardous', axis=1)
y = df_clean['hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print(f"Rozmiar zbioru treningowego: {X_train_scaled.shape[0]} probek")
print(f"Rozmiar zbioru testowego: {X_test_scaled.shape[0]} probek")