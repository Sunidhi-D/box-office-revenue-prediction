import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipelinef 
rom sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import  r2_score, mean_absolute_error 

file_path=r'\Movie_regression.csv‘ 
df = pd.read_csv (file_path, encoding='latin-1') 
df.head () 
df=df.dropna() 
df.info () 

df.hist (bins = 15, color = "r", edgecolor = "black", xlabelsize = 8, ylabelsize = 8, linewidth = 1, grid = False) 
plt.tight_layout (rect = (0,0, 1.2,1.2)) 
plt.suptitle ("Movie Regression Univariate analysis", x = 0.65, y = 1.25) 
plt.show () 

label_encoder = LabelEncoder() 
label= label_encoder.fit_transform(df['Genre']) 
df['Genre']=label 
df.head() 

corr = df.corr () 

plt.figure (figsize = (12, 10)) 
sns.heatmap (corr, annot = True, fmt = ".2f", cmap = "coolwarm_r", linewidth = 0.2) 
plt.title ("movie regression multivariate plot", fontsize = 14) 
plt.show () 

plt.scatter (df["Collection"], df["Budget"], color = "red") 
plt.xlabel ("Budget") 
plt.ylabel ("Collection") 
plt.show () 

plt.scatter (df["Collection"], df["Trailer_views"], color = "red") 
plt.xlabel ("Trailer_views") 
plt.ylabel ("Collection") 
plt.show () 

sns.boxplot (x = "Genre", y = "Collection", data = df, palette = "colorblind") 

X=df[['Marketing expense','Production expense','Multiplex coverage','Budget','Movie_length','Lead_ 
Actor_Rating','Lead_Actress_rating','Director_rating','Producer_rating','Critic_rating','Trailer_views','Time_taken','Twitter_hastags','Genre','Avg
 _age_actors','Num_multiplex']] 
Y=df[['Collection']] 

numeric_features=['Marketing expense','Production expense','Multiplex coverage','Budget','Movie_length','Lead_ 
Actor_Rating','Lead_Actress_rating','Director_rating','Producer_rating','Critic_rating','Trailer_views','Time_taken','Twitter_hastags','Genre','Avg
 _age_actors','Num_multiplex'] 
categorical_features=['Genre'] 

df['Genre']=label_encoder.inverse_transform(label) 
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())]) 
categorical_transformer =Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
preprocessor=ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ('cat', categorical_transformer, 
categorical_features)]) 
model=Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())]) 

X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.2,random_state=42) 

model.fit(X_train, y_train) 
y_pred=model.predict(X_test) 

y_test = np.array(y_test, dtype=float) 
y_pred = np.array(y_pred, dtype=float) 
mean=mean_absolute_error(y_test, y_pred) 
r2=r2_score(y_test, y_pred) 
print(f'R^2 Score: {r2}') 
print(f'mean absolute error: {mean}') 

plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred, alpha=0.7, color='b') 
plt.xlabel('Actual Collection') 
plt.ylabel('Predicted Collection') 
plt.title('Actual vs Predicted Collection') 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],color='red', linestyle='-', linewidth=2, label='Regression Line') 
plt.legend() 
plt.show() 

residuals=y_test-y_pred 
plt.figure(figsize=(10, 6)) 
plt.scatter(y_pred, residuals, alpha=0.7, color='b') 
plt.xlabel('Predicted Collection') 
plt.ylabel('Residuals') 
plt.title("Residuals vs Predicted Collection") 
plt.axhline(y=0, color='red') 
plt.show()
