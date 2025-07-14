# box-office-revenue-prediction

# 🎬 Box Office Revenue Prediction using Linear Regression

This project demonstrates how to use machine learning (Linear Regression) to predict the box office revenue of movies based on historical features, using a public dataset from Keras.

---

## 📁 Dataset

- **Source**: Preloaded dataset from `keras.datasets` (or a similar format)
- Includes features such as:
  - Budget
  - Actor and Director Popularity
  - Marketing Spend
  - Trailer Views
  - Genre

---

## 🛠 Libraries Used

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## 🔧 Workflow

1. Data loading from Keras  
2. Data Cleaning and Preprocessing  
3. Label encoding & feature scaling  
4. Linear Regression model training  
5. Evaluation using R² score and MAE  

---

## 📊 Sample Results

Input:
{
  "Budget": ₹50,00,00,000,
  "Marketing Expense": ₹10,00,00,000,
  "Trailer Views": 5,000,000,
  "Actor Rating": 8.5,
  "Director Rating": 9.0,
  "Genre": "Action"
}

Output:
Predicted Box Office Revenue: 62.4 Crores


---

## 🔮 Future Improvements

- Try other regressors (SVR, Ridge, Lasso)  
- Add outlier detection  
- Build a simple web interface using Streamlit

---

## 👩‍💻 Author

Sunidhi Divekar  
[GitHub](https://github.com/Sunidhi-D)
