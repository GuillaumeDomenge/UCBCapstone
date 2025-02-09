import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

# load data
df = pd.read_csv('./Dataset/Processed Data/test_set_cpc_5scores.csv', sep=',')
df["Num_Im_Diff"] = np.abs(df["Num_Im_1"]-df["Num_Im_2"])
df["Num_Im_Sum"] = np.abs(df["Num_Im_1"]+df["Num_Im_2"])

#feats = ["SIFT_top1","SIFT_top2","SIFT_top3","SIFT_avg","SIFT_std","SIFT_worse","Num_Im_1","Num_Im_2","Num_Im_Diff","Num_Im_Sum"]
#feats = ["Num_Im_1","Num_Im_2","Num_Im_Diff","Num_Im_Sum"]
feats = ["SIFT_top1","SIFT_top2","SIFT_top3","SIFT_avg","SIFT_std","SIFT_worse"]
feats.append("is_same")
selected_columns = df[feats]

#transfer to numpy
numpy_array = selected_columns.to_numpy()
print(f'The size of the dataset:{numpy_array.shape}')
X_train, X_test, y_train, y_test = train_test_split(numpy_array[:, :-1], numpy_array[:, -1], test_size=0.2, random_state=157190)

# train a SVC and a RF
print(f'Training on a dataset of {X_train.shape}')

def experiment(model, X_train,y_train,X_test,y_test):
    model.fit(X_train, y_train)
    hat_y = model.predict(X_test)
    cm = confusion_matrix(y_test, hat_y)
    true_positive_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print(f'Testing ACC: {accuracy_score(y_test,hat_y) * 100:.0f}%, TPR:{true_positive_rate * 100:.0f}%, FPR:{false_positive_rate * 100:.0f}%, F1:{f1_score(y_test,hat_y) * 100:.0f}%, AU_ROC{roc_auc_score(y_test,hat_y) * 100:.0f}%')

svm = SVC(random_state=157190)
rf = RandomForestClassifier(random_state=157190)
mlp = MLPClassifier(random_state=157190)

assert max(y_test)==1 and min(y_test)==0
print("Build with SVM")
experiment(svm,X_train,y_train,X_test,y_test)
print("Build with RF")
experiment(rf,X_train,y_train,X_test,y_test)
print("Build with MLP")
experiment(mlp,X_train,y_train,X_test,y_test)

