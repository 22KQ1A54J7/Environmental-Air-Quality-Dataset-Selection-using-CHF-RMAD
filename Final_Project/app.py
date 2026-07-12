import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from flask import Flask, render_template, request, flash, send_from_directory
from fcode.main import run_chf_rough_madm

app = Flask(__name__)
app.secret_key = "air_quality_secret"

UPLOAD_FOLDER = "datasets"
RESULT_FOLDER = "results"
CHART_FOLDER = os.path.join("static", "charts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# Example weights
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# -------------------------------
# Detect Dataset Purpose
# -------------------------------
def detect_dataset_purpose(df):
    cols = [c.lower() for c in df.columns]

    if "rh" in cols and "ah" in cols:
        return "Policy Making & Health Impact Analysis"
    if "temperature" in cols and "humidity" in cols:
        return "Real-time Monitoring & Smart City Applications"
    if "latitude" in cols and "longitude" in cols:
        return "Large-scale Regional & Climate Analysis"
    if "o3" in cols and "aqi" in cols:
        return "City Comparison & Public Reporting"
    if "hospital" in cols or "asthma" in cols:
        return "Health Risk & Epidemiological Studies"

    return "General Air Quality Analysis"

# -------------------------------
# Home Page
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    final_csv_path = os.path.join(RESULT_FOLDER, "final_dataset_ranking.csv")

    # Load previous results if exist
    if os.path.exists(final_csv_path):
        prev_df = pd.read_csv(final_csv_path)
        dataset_results = prev_df.to_dict(orient="records")
    else:
        dataset_results = []

    if request.method == "POST":
        files = request.files.getlist("files")

        existing_names = [d["Dataset"] for d in dataset_results]

        for file in files:
            filename = file.filename

            # Only allow CSV
            if not filename.lower().endswith(".csv"):
                flash(f"Only CSV files allowed: {filename}", "error")
                continue

            # Prevent duplicate dataset name
            if filename in existing_names:
                flash(f"Dataset already exists: {filename}", "warning")
                continue

            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            df = pd.read_csv(save_path)

            purpose = detect_dataset_purpose(df)

            # Run MADM
            result_df, best_score = run_chf_rough_madm(save_path, weights)

            dataset_results.append({
                "Dataset": filename,
                "Best_Score": float(best_score),
                "Purpose": purpose
            })

        # Create ranking
        if dataset_results:
            ranking_df = pd.DataFrame(dataset_results)
            ranking_df = ranking_df.sort_values(by="Best_Score", ascending=False)
            ranking_df["Dataset_Rank"] = range(1, len(ranking_df) + 1)

            # Save final CSV
            ranking_df.to_csv(final_csv_path, index=False)

            # Create charts
            create_bar_chart(ranking_df)
            create_pie_chart(ranking_df)

            dataset_ranking = ranking_df.to_dict(orient="records")
        else:
            dataset_ranking = []

    else:
        if os.path.exists(final_csv_path):
            ranking_df = pd.read_csv(final_csv_path)
            dataset_ranking = ranking_df.to_dict(orient="records")
        else:
            dataset_ranking = []

    return render_template(
        "index.html",
        dataset_ranking=dataset_ranking,
        bar_chart=os.path.exists(os.path.join(CHART_FOLDER, "dataset_bar_chart.png")),
        pie_chart=os.path.exists(os.path.join(CHART_FOLDER, "dataset_pie_chart.png"))
    )

# -------------------------------
# Download Final CSV
# -------------------------------
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


def create_bar_chart(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Dataset"], df["Best_Score"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Dataset Score Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, "dataset_bar_chart.png"))
    plt.close()

# -------------------------------
# Pie Chart
# -------------------------------
def create_pie_chart(df):
    plt.figure(figsize=(7, 7))
    plt.pie(df["Best_Score"], labels=df["Dataset"], autopct="%1.1f%%", startangle=140)
    plt.title("Dataset Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, "dataset_pie_chart.png"))
    plt.close()

# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
