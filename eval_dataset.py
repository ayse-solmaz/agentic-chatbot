from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "agentic-chatbot-rag-qa"

# questions and reference answers written from my_paper.pdf
# ("Development of Multiple Combined Regression Methods for Rainfall Measurement")
EXAMPLES = [
    {
        "question": "Which regression model performed best in the paper and what were its scores?",
        "answer": "Random Forest regressor performed best with an R2 score of 0.869904217, "
                  "MAE of 0.194459262, MSE of 0.126358647 and RMSE of 0.355469615.",
    },
    {
        "question": "How many supervised regressors were trained and compared?",
        "answer": "10 supervised regressors: Linear, Polynomial, Ridge, Lasso, Decision Tree, "
                  "KNN, Random Forest, AdaBoost, Gradient Boosting, XGBoost (plus SVM is also reported).",
    },
    {
        "question": "Where does the dataset used in the paper come from?",
        "answer": "From Kaggle (the weather dataset from the Rattle package): about 10 years of daily "
                  "weather observations from many locations in Australia.",
    },
    {
        "question": "Which models came second and third by R2 score?",
        "answer": "Gradient Boosting Regressor was second with R2 0.863496747 and "
                  "XGBoost was third with R2 0.863215393.",
    },
    {
        "question": "Makalede aykırı değerler (outliers) hangi yöntemle ele alındı?",
        "answer": "IQR (interquartile range / çeyrekler açıklığı) yöntemi kullanıldı.",
    },
    {
        "question": "Which feature scaling and categorical encoding methods were used?",
        "answer": "Standard scaling (StandardScaler) for feature scaling and one-hot encoding "
                  "for categorical features.",
    },
    {
        "question": "How were missing values handled?",
        "answer": "Rows and columns with more than 50% null values were removed, and the remaining "
                  "missing values were filled with the mean, median and mode strategy.",
    },
    {
        "question": "Which model performed worst and what was its R2 score?",
        "answer": "Lasso Regression performed worst with an R2 score of about -5.91E-05 "
                  "(RMSE 0.985561434).",
    },
    {
        "question": "What were the KNN regressor's settings and R2 score?",
        "answer": "KNN used 5 nearest neighbors with Manhattan distance and got an R2 score of 0.401557082.",
    },
    {
        "question": "Which evaluation metrics were used to compare the models?",
        "answer": "R2 score, mean absolute error (MAE), mean squared error (MSE) "
                  "and root mean squared error (RMSE).",
    },
    {
        "question": "Yazarlar gelecekte ne yapmayı planlıyor?",
        "answer": "Çiftçilerin kolayca kullanabileceği gerçek hayatta bir yağış tahmin uygulaması geliştirmek, "
                  "performansı artırmak için sinir ağı tabanlı derin hibrit yöntemler kullanmak ve "
                  "başka ülkelerin verileriyle tahmin yapmak.",
    },
    {
        "question": "Which tools and libraries were used to run the experiments?",
        "answer": "Google Colab as the environment, scikit-learn for the regression models, "
                  "and matplotlib and seaborn for visualization.",
    },
]


def main():
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists, skipping.")
        return

    dataset = client.create_dataset(
        DATASET_NAME,
        description="Questions about my_paper.pdf with reference answers, for evaluating the RAG chatbot.",
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": example["question"]},
                "outputs": {"answer": example["answer"], "expected_tool": "rag_tool"},
            }
            for example in EXAMPLES
        ],
    )

    print(f"Created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    main()
