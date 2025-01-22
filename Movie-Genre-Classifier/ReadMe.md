# Movie Genre Classification

This project involves creating a machine learning model to classify movies into genres based on their plot summaries. The model utilizes techniques like TF-IDF for text vectorization and Logistic Regression for classification.

## Features

- **Automated Dataset Download**: The dataset is downloaded directly from Kaggle using `kagglehub`.
- **Text Preprocessing**: Includes TF-IDF vectorization to convert textual data into numerical features.
- **Classification**: Implements Logistic Regression to predict movie genres.
- **Model Evaluation**: Includes accuracy, classification report, and confusion matrix metrics.
- **Model Persistence**: Saves the trained model and vectorizer for reuse.

## Dataset

The dataset used for this project is the [IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb). It contains movie plot summaries and their corresponding genres.

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Libraries: 
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - kagglehub

Install dependencies using:
```bash
pip install pandas numpy scikit-learn joblib kagglehub
```

## Setup

### 1. Kaggle API Configuration

- Download your Kaggle API token (`kaggle.json`) from the Kaggle website.
- Place it in the directory: `~/.kaggle/kaggle.json`.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification
```

### 3. Run the Script

Execute the Python script to download the dataset, train the model, and evaluate it:

```bash
python movie_genre_classification.py
```

## Project Structure

```
├── README.md                # Project documentation
├── movie_genre_classification.py  # Main Python script
├── requirements.txt         # List of dependencies
└── saved_models/            # Directory for saved models and vectorizers
```

## Usage

### Training the Model
Run the script to train and evaluate the model:
```bash
python movie_genre_classification.py
```

### Predicting a Genre
After training, you can use the saved model for predictions:

```python
from joblib import load

# Load the saved model and vectorizer
model = load('saved_models/movie_genre_model.pkl')
vectorizer = load('saved_models/tfidf_vectorizer.pkl')

# Example prediction
plot_summary = ["A young wizard discovers his magical heritage."]
features = vectorizer.transform(plot_summary)
predicted_genre = model.predict(features)
print("Predicted Genre:", predicted_genre)
```

## Results

The model achieves an accuracy of ~XX% on the test dataset. Detailed performance metrics are available in the console output.

## Contribution

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for providing the dataset.
- OpenAI for their valuable resources on machine learning.

---

Happy coding! 🎉
