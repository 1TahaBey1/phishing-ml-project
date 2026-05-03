import pandas as pd


df = pd.read_csv("dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df["Result"].value_counts())

# index sütununu çıkardık
df = df.drop("index", axis=1)

# Label düzeltme
# Eski -1 = phishing, 1 = legitimate
# Yeni: 0 = phishing, 1 = legitimate
df["Result"] = df["Result"].replace(-1, 0)

print("After label conversion:")
print(df["Result"].value_counts())

# 5) X ve y ayır
X = df.drop("Result", axis=1)
y = df["Result"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

#Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

#k-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

#Sonuçları değerlendirme
from sklearn.metrics import accuracy_score
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Naive Bayes:", accuracy_score(y_test, nb_pred))
print("k-NN:", accuracy_score(y_test, knn_pred))



#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Decision Tree Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.show()

#K-NN Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt="d")
plt.title("k-NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm_nb, annot=True, fmt="d")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#Metrikler
from sklearn.metrics import classification_report
print("Decision Tree:\n", classification_report(y_test, dt_pred))
print("Naive Bayes:\n", classification_report(y_test, nb_pred))
print("k-NN:\n", classification_report(y_test, knn_pred))