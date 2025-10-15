
#create a tensor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy


device = torch.device("cpu")  # فقط CPU

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc



# ساخت داده‌ها
NUM_CLASSSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. ساخت داده‌های چندکلاسه
X_blob, Y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSSES, cluster_std=1.5, random_state=RANDOM_SEED)

# 2. تبدیل داده‌ها به تنسور
X_blob = torch.from_numpy(X_blob).type(torch.float)
Y_blob = torch.from_numpy(Y_blob).type(torch.long)  # برای استفاده در CrossEntropy باید Y_blob به long تبدیل بشه

# 3. تقسیم داده‌ها به مجموعه آموزش و تست
X_blob_train, X_blob_test, Y_blob_train, Y_blob_test = train_test_split(X_blob, Y_blob, test_size=0.2, random_state=RANDOM_SEED)

# 4. نمایش داده‌ها
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=Y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# مدل برای دسته‌بندی چندکلاسه
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=128):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

# ایجاد مدل
model_4 = BlobModel(input_features=2, output_features=4, hidden_units=8).to(device)

# تابع خطا برای دسته‌بندی چندکلاسه
loss_fn = nn.CrossEntropyLoss()

# بهینه‌ساز
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)

# تنظیم مقدار اولیه برای دستگاه
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ارسال داده‌ها به دستگاه
X_blob_train, Y_blob_train = X_blob_train.to(device), Y_blob_train.to(device)
X_blob_test, Y_blob_test = X_blob_test.to(device), Y_blob_test.to(device)

# آموزش مدل
epochs = 100

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)  # پیش‌بینی مدل
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)  # پیش‌بینی‌ها را به برچسب‌ها تبدیل کن

    loss = loss_fn(y_logits, Y_blob_train)
    acc = accuracy_fn(y_true=Y_blob_train, y_pred=y_preds)  # دقت مدل

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ارزیابی مدل
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)  # پیش‌بینی مدل
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)  # انتخاب بیشترین احتمال
        test_loss = loss_fn(test_logits, Y_blob_test)  # محاسبه‌ی خطا
        test_acc = accuracy_fn(y_true=Y_blob_test, y_pred=test_preds)  # محاسبه‌ی دقت (accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # setup metric (TorchMetrics)
    # torchmetric_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
    # n = torchmetric_accuracy(test_preds, Y_blob_test)
    # print(f"TorchMetrics Accuracy: {n * 100:.2f}%")
