import pandas as pd


def evaluate_recommendations():
    df = pd.read_csv("outputs/recommendations.csv")

    report = pd.DataFrame({
        "Metric": [
            "Total Recommendations",
            "Average Recommendation Score",
            "Highest Recommendation Score",
            "Lowest Recommendation Score",
            "Average Traffic Score",
            "Average Road Connectivity",
            "Average Mall Score",
            "Average Hospital Score",
            "Average Metro Score"
        ],
        "Value": [
            len(df),
            df["RecommendationScore"].mean(),
            df["RecommendationScore"].max(),
            df["RecommendationScore"].min(),
            df["TrafficScore"].mean(),
            df["RoadConnectivity"].mean(),
            df["MallScore"].mean(),
            df["HospitalScore"].mean(),
            df["MetroScore"].mean()
        ]
    })

    report.to_csv("outputs/evaluation_report.csv", index=False)

    print(report)
    print("\nEvaluation report saved to outputs/evaluation_report.csv")


if __name__ == "__main__":
    evaluate_recommendations()