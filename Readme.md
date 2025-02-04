Fake News Detection using Machine Learning
Overview
This project implements a Fake News Detection system using Machine Learning algorithms to classify news articles as fake or real. The system leverages Natural Language Processing (NLP) techniques, and the performance of two classifiers—Random Forest and Naïve Bayes—is compared.

- Random Forest Classifier: 99% accuracy
- Naïve Bayes Classifier: 93.8% accuracy

The goal is to demonstrate how machine learning can be used to detect fake news in a text dataset.
Features
- Data Preprocessing: Utilizes TF-IDF vectorization for transforming text data.
- Model Comparison: Compares the performance of Random Forest and Naïve Bayes classifiers.
- Visualizations: Provides insights into the data and model performance using Matplotlib and Seaborn.
- Accuracy: Achieves high accuracy with Random Forest surpassing Naïve Bayes.
Technologies Used
- Python 🐍
- Scikit-learn (Machine Learning)
- Pandas (Data Handling)
- NumPy (Mathematical Operations)
- TF-IDF (Text Feature Extraction)
- Matplotlib & Seaborn (Data Visualization)
- Natural Language Processing (NLP)
Installation
### Clone the repository
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Install dependencies
Make sure you have all required libraries installed. You can use `pip` to install them from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Run the code
Once the dependencies are installed, you can run the main code file to train and evaluate the models:
```bash
python fake_news_detection.py
```
Dataset
The project uses a publicly available fake news dataset, which contains labeled real and fake news articles. The dataset is available at Kaggle and can be downloaded and placed in the appropriate folder in the repository.
[Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
Results
- The Random Forest Classifier achieves 99% accuracy, making it the better-performing model for this task.
- Naïve Bayes achieves 93.8% accuracy, providing an alternative approach for classification.

Visualizations and detailed comparisons between both models are provided in the code output.
Contributing
Contributions to this project are welcome! You can submit issues, suggest improvements, or contribute new features through pull requests.

### How to Contribute
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to your forked repository (`git push origin feature-name`).
6. Open a pull request.
Acknowledgments
- Kaggle for providing the dataset used in this project.
- Scikit-learn for the machine learning algorithms.
- Pandas and NumPy for data manipulation and processing.
