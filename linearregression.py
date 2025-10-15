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

# اگر توی همین فایل داری اجرا می‌کنی نیازی به ایمپورت device نیست
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ایجاد داده‌های خطی برای تمرین مدل رگرسیون
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# داده‌های ورودی X و خروجی Y (یعنی y = 0.7x + 0.3)
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
Y_regression = weight * X_regression + bias

# تقسیم داده‌ها به train و test
train_split = int(0.8 * len(X_regression))
X_train_regression, Y_train_regression = X_regression[:train_split], Y_regression[:train_split]
X_test_regression, Y_test_regression = X_regression[train_split:], Y_regression[train_split:]

print(len(X_train_regression), len(Y_train_regression))
print(len(X_test_regression), len(Y_test_regression))

# --- تابع برای رسم داده‌ها و پیش‌بینی‌ها ---
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend()
    plt.show()

# نمایش داده‌های اولیه
plot_predictions(X_train_regression, Y_train_regression, X_test_regression, Y_test_regression)

# تعریف مدل چند لایه (MLP ساده)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# تابع خطا و بهینه‌ساز
loss_fn = nn.L1Loss()  # خطای میانگین قدرمطلق
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

# آماده‌سازی داده‌ها روی CPU یا GPU
X_train_regression, Y_train_regression = X_train_regression.to(device), Y_train_regression.to(device)
X_test_regression, Y_test_regression = X_test_regression.to(device), Y_test_regression.to(device)

# آموزش مدل
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    # حالت آموزش
    model_2.train()

    # پیش‌بینی روی داده‌های آموزش
    y_pred = model_2(X_train_regression)

    # محاسبه خطا
    loss = loss_fn(y_pred, Y_train_regression)

    # به‌روزرسانی وزن‌ها
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # حالت ارزیابی
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, Y_test_regression)

    # چاپ هر 100 epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")

# پیش‌بینی نهایی و رسم نمودار
model_2.eval()
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

plot_predictions(X_train_regression.cpu(), Y_train_regression.cpu(),
                 X_test_regression.cpu(), Y_test_regression.cpu(),
                 predictions=y_preds.cpu())


