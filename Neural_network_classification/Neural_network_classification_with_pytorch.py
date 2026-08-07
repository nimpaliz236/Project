# ------------------------------ کتابخانه‌ها ------------------------------
import sklearn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from torch import nn
import requests
from pathlib import Path
from helper_functions import plot_predictions, plot_decision_boundary

# ------------------------------ ساخت داده دایره‌ای ------------------------------
n_samples = 1000  # تعداد نمونه‌ها
X, y = make_circles(n_samples=n_samples, noise=0.2, random_state=42)  # ایجاد داده‌های دایره‌ای با نویز

# ساخت دیتافریم برای دیدن داده‌ها
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

# رسم داده‌ها برای مشاهده
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("Circle Dataset Visualization")
plt.show()

# ------------------------------ تبدیل داده‌ها به Tensor ------------------------------
x = torch.from_numpy(X).type(torch.float)  # تبدیل ویژگی‌ها به تنسور
y = torch.from_numpy(y).type(torch.float)  # تبدیل برچسب‌ها به تنسور

torch.manual_seed(42)  # برای قابل تکرار بودن نتایج

# ------------------------------ تقسیم داده‌ها ------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ------------------------------ انتخاب Device (CPU) ------------------------------
device = torch.device("cpu")
print("Device:", device)

# ------------------------------ تعریف مدل ------------------------------
# روش اول: با کلاس
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)  # لایه اول با ۲ ورودی و ۵ خروجی
        self.layer2 = nn.Linear(in_features=5, out_features=1)  # لایه دوم با ۵ ورودی و ۱ خروجی

    def forward(self, x):
        return self.layer2(self.layer1(x))  # عبور داده از لایه‌ها

# نمونه‌سازی از مدل
model_0_class = CircleModelV0().to(device)

# روش دوم: با nn.Sequential (مدل کوتاه‌تر)
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=128),  # لایه ورودی با ۲ ویژگی و ۱۲۸ نرون
    nn.Linear(in_features=128, out_features=1),  # لایه خروجی با ۱ نرون (برای خروجی باینری)
).to(device)

# ------------------------------ پیش‌بینی اولیه (قبل از آموزش) ------------------------------
with torch.inference_mode():  # غیرفعال کردن گرادیان برای پیش‌بینی
    untrained_preds = model_0(X_test.to(device))

# ------------------------------ تعریف تابع خطا و بهینه‌ساز ------------------------------
loss_fn = nn.BCEWithLogitsLoss()  # ترکیب سیگموید و CrossEntropy برای کلاس‌بندی دودویی
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)  # الگوریتم گرادیان نزولی

# ------------------------------ تابع دقت ------------------------------
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_pred, y_true).sum().item()  # شمارش نمونه‌های درست
    acc = (correct / len(y_true)) * 100  # درصد درستی
    return acc

# ------------------------------ بررسی اولیه پیش‌بینی‌ها ------------------------------
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
    print("Logits:\n", y_logits)  # خروجی خام مدل (قبل از سیگموید)

# اعمال تابع سیگموید روی خروجی برای تبدیل به احتمال
y_pred_probs = torch.sigmoid(y_logits)

# گرد کردن خروجی به ۰ و ۱ (برچسب نهایی)
y_pred_labels = torch.round(y_pred_probs)

# مقایسه برچسب‌های واقعی و پیش‌بینی‌شده
print("Predictions vs Labels Equality: ",
      torch.eq(y_pred_labels.squeeze(), y_test[:5].to(device).squeeze()))
print("Predicted Labels:", y_pred_labels.squeeze())
print("True Labels:", y_test[:5].to(device).squeeze())

# ------------------------------ آماده‌سازی برای آموزش ------------------------------
torch.manual_seed(42)
epochs = 100  # تعداد دورهای آموزش

# ارسال داده‌ها به دستگاه مناسب (CPU)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ------------------------------ حلقه آموزش مدل ------------------------------
for epoch in range(epochs):
    model_0.train()  # مدل در حالت آموزش

    # عبور داده‌ها از مدل (forward pass)
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # تبدیل خروجی به ۰ یا ۱

    # محاسبه خطا و دقت
    loss = loss_fn(y_logits, y_train)  # تابع خطا با خروجی خام (logits)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # صفر کردن گرادیان‌ها
    optimizer.zero_grad()

    # محاسبه گرادیان‌ها (backward)
    loss.backward()

    # به‌روزرسانی وزن‌ها
    optimizer.step()

    # ------------------------------ ارزیابی مدل روی داده تست ------------------------------
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # چاپ وضعیت هر ۱۰ epoch
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Train acc: {acc:.2f}% "
              f"| Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

    # رسم مرز تصمیم‌گیری
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train)

    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_test, y_test)

    plt.show()  # نمایش نمودارها

    class CirclemodelV1(nn.Module):
     def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)

        def forward(self, x):
            # z=self.layer1(x)
            # z=self.layer2(z)
            # z=self.layer3(z)
            return self.layer3(self.layer2(self.layer1(x)))

#--------------------------گسترش مدل-----------------------------------------
#
# model_1=CirclemodelV1().to(device)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# epochs = 1000
# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)
# for epoch in range(epochs):
#     model_1.train()
#     y_logits = model_1(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits))
#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     model_1.eval()
#     with torch.inference_mode():
#         test_logits = model_1(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
#









