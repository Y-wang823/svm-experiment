# -*- coding:utf-8 -*-
"""
SVM垃圾邮件分类实验 Spambase数据集
参考课件案例①：垃圾邮件分类
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ========== 1.加载数据集 ==========
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
col_names = [f"feat_{i}" for i in range(57)] + ["is_spam"]
df = pd.read_csv(url, names=col_names)

X = df.drop("is_spam", axis=1)
y = df["is_spam"]

print(f"总样本数：{df.shape[0]},特征维度：{X.shape[1]}")
print(f"垃圾邮件占比：{y.mean():.2%}")

# ========== 2.划分数据集 & 标准化（SVM必须！课件重点） ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========3. 模型训练 ==========
## ①线性SVM
linear_svm = LinearSVC(C=1, max_iter=10000, random_state=42)
linear_svm.fit(X_train_scaled, y_train)
y_pred_linear = linear_svm.predict(X_test_scaled)

## ②RBF核SVM，网格搜索调参C、gamma
param_grid = {
    "C": [1, 10],
    "gamma": [0.01, 0.1]
}
rbf_svm = SVC(kernel="rbf")
grid = GridSearchCV(rbf_svm, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_rbf = grid.best_estimator_
y_pred_rbf = best_rbf.predict(X_test_scaled)

## ③逻辑回归baseline
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# ==========4.模型评估 ==========
def show_result(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print("-"*50)
    print(f"【{name}】准确率 = {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=3))
    return acc

acc_linear = show_result("线性SVM C=1", y_test, y_pred_linear)
acc_rbf = show_result(f"RBF SVM best params:{grid.best_params_}", y_test, y_pred_rbf)
acc_lr = show_result("逻辑回归", y_test, y_pred_lr)

print(f"\nRBF SVM支持向量数目：{best_rbf.n_support_}")

result_df = pd.DataFrame({
    "模型":["线性SVM(C=1)","RBF‑SVM","逻辑回归"],
    "测试集准确率":[acc_linear, acc_rbf, acc_lr]
})
print("\n=====结果汇总=====")
print(result_df)

# ==========5.超参数C影响绘图==========
c_list = [0.001,0.01,0.1,1,10,100]
acc_list = []
for c in c_list:
    clf = LinearSVC(C=c, max_iter=20000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    pred = clf.predict(X_test_scaled)
    acc_list.append(accuracy_score(y_test,pred))

plt.figure(figsize=(8,4))
plt.plot(c_list,acc_list,marker='o')
plt.xscale("log")
plt.xlabel("C(惩罚系数，对数刻度)")
plt.ylabel("测试集准确率")
plt.title("软间隔SVM：超参数C对模型性能影响")
plt.grid(True,alpha=0.4)
plt.savefig("c_performance.png")
plt.show()
