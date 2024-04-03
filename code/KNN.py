import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")
data = pd.read_csv("forest_fires.csv")
data = data.dropna()
X = data[['Temperature','Humidity','Oxygen']].values
y = data['Fire Occurrence'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)
pickle.dump(knn, open('model.pkl','wb'))