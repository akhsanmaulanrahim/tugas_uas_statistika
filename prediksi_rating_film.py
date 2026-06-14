from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "movie_rating_dataset_100(1).csv"
OUTPUT_DIR = BASE_DIR / "hasil_visualisasi"


def build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    required_columns = ["Duration_Minutes", "Genre", "Votes", "Rating"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Kolom dataset belum lengkap: {missing_columns}")

    return data


def create_model():
    numeric_features = ["Duration_Minutes", "Votes"]
    categorical_features = ["Genre"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def save_visualizations(data, y_test, y_pred, model):
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(data["Rating"], bins=10, color="#2f80ed", edgecolor="white")
    plt.title("Distribusi Rating Film")
    plt.xlabel("Rating")
    plt.ylabel("Jumlah Film")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distribusi_rating.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Duration_Minutes"], data["Rating"], alpha=0.75, color="#27ae60")
    plt.title("Hubungan Durasi Film dan Rating")
    plt.xlabel("Durasi Film (menit)")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_durasi_vs_rating.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["Votes"], data["Rating"], alpha=0.75, color="#f2994a")
    plt.title("Hubungan Jumlah Voting dan Rating")
    plt.xlabel("Jumlah Voting")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_votes_vs_rating.png", dpi=150)
    plt.close()

    genre_rating = data.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    genre_rating.plot(kind="bar", color="#9b51e0")
    plt.title("Rata-rata Rating Berdasarkan Genre")
    plt.xlabel("Genre")
    plt.ylabel("Rata-rata Rating")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_rata_rating_genre.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color="#eb5757")
    min_rating = min(y_test.min(), y_pred.min())
    max_rating = max(y_test.max(), y_pred.max())
    plt.plot([min_rating, max_rating], [min_rating, max_rating], color="black", linestyle="--")
    plt.title("Rating Aktual vs Prediksi")
    plt.xlabel("Rating Aktual")
    plt.ylabel("Rating Prediksi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_aktual_vs_prediksi.png", dpi=150)
    plt.close()

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    importance = pd.Series(np.abs(coefficients), index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    importance.plot(kind="barh", color="#56ccf2")
    plt.title("Pengaruh Fitur pada Model Regresi")
    plt.xlabel("Nilai Absolut Koefisien")
    plt.ylabel("Fitur")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_pengaruh_fitur.png", dpi=150)
    plt.close()


def predict_new_movie(model, duration_minutes, genre, votes):
    new_data = pd.DataFrame(
        {
            "Duration_Minutes": [duration_minutes],
            "Genre": [genre],
            "Votes": [votes],
        }
    )
    predicted_rating = model.predict(new_data)[0]
    return round(float(predicted_rating), 2)


def main():
    data = load_dataset()

    print("=== DATASET FILM ===")
    print(data.head())
    print("\nJumlah data:", len(data))
    print("\nGenre tersedia:", ", ".join(sorted(data["Genre"].dropna().unique())))

    x = data[["Duration_Minutes", "Genre", "Votes"]]
    y = data["Rating"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = create_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== EVALUASI MODEL REGRESI LINEAR ===")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R2): {r2:.3f}")

    comparison = pd.DataFrame(
        {
            "Rating_Aktual": y_test.values,
            "Rating_Prediksi": np.round(y_pred, 2),
        }
    )
    print("\n=== PERBANDINGAN RATING AKTUAL DAN PREDIKSI ===")
    print(comparison.head(10))

    save_visualizations(data, y_test, y_pred, model)
    print(f"\nVisualisasi berhasil disimpan di folder: {OUTPUT_DIR}")

    example_prediction = predict_new_movie(
        model=model,
        duration_minutes=140,
        genre="Drama",
        votes=250000,
    )
    print("\n=== CONTOH PREDIKSI FILM BARU ===")
    print("Durasi: 140 menit | Genre: Drama | Votes: 250000")
    print(f"Prediksi rating: {example_prediction}")


if __name__ == "__main__":
    main()
