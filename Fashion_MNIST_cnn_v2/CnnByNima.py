# -------------------- کتابخانه‌های لازم --------------------
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import random
import mlxtend
from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF


# -------------------- انتخاب دستگاه اجرا --------------------
device = torch.device("cpu")

# -------------------- بارگذاری داده‌ها --------------------
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- نام کلاس‌ها --------------------
class_names = train_data.classes

# -------------------- تعریف مدل CNN --------------------
class FashionMNISTmodelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# -------------------- ساخت مدل --------------------
torch.manual_seed(42)
model_2 = FashionMNISTmodelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

# -------------------- تابع ارزیابی مدل --------------------
def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, accuracy_fn, device=device):
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, Y)
            acc += accuracy_fn(y_true=Y, y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, "model_loss": loss.item(), "model_accuracy": acc}

# -------------------- تابع نمایش زمان آموزش --------------------
def print_train_time(start: float, end: float, device: torch.device = None):
    total = end - start
    print(f"\n⏱️ زمان کل آموزش روی {device}: {total:.3f} ثانیه")
    return total

# -------------------- مرحله‌ی آموزش --------------------
def train_step(model, dataloader, loss_fn, optimizer, accuracy, device):
    model.train()
    train_loss, train_acc = 0, 0
    progress_bar = tqdm(dataloader, desc="Training", leave=True)
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += accuracy(y_true=y, y_pred=y_pred_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_acc/(batch+1):.4f}"})
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"\n✅ Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")

# -------------------- مرحله‌ی تست --------------------
def test_step(model, data_loader, loss_fn, accuracy, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        progress_bar = tqdm(data_loader, desc="Testing", leave=False)
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_pred_class = test_pred.argmax(dim=1)
            test_acc += accuracy(y_true=y, y_pred=test_pred_class)
            progress_bar.set_postfix({
                "loss": f"{test_loss / (len(progress_bar) + 1):.4f}",
                "acc": f"{test_acc / (len(progress_bar) + 1):.4f}"
            })
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"\n🧪 Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}\n")

# -------------------- تنظیمات آموزش --------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)
epochs = 3

# -------------------- حلقه‌ی آموزش --------------------
train_time_start = timer()
for epoch in range(epochs):
    print(f"\n📘 Epoch {epoch + 1}/{epochs}")
    train_step(model_2, train_loader, loss_fn, optimizer, accuracy_fn, device)
    test_step(model_2, test_loader, loss_fn, accuracy_fn, device)
train_time_end = timer()
print_train_time(train_time_start, train_time_end, device)

# -------------------- ارزیابی نهایی مدل --------------------
model_2_results = eval_model(model_2, test_loader, loss_fn, accuracy_fn, device)
print(model_2_results)

compare_results = pd.DataFrame([model_2_results])
print(compare_results)

# -------------------- تابع پیش‌بینی روی چند تصویر --------------------
def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample.to(device))
    test_labels.append(torch.tensor(label).to(device))

pred_probs = make_predictions(model_2, test_samples, device)
pred_classes = torch.argmax(pred_probs, dim=1)

plt.figure(figsize=(9, 9))
nrows, ncols = 3, 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i + 1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    color = "g" if pred_label == truth_label else "r"
    plt.title(f"pred: {pred_label}, truth: {truth_label}", fontsize=10, color=color)
    plt.axis(False)
plt.tight_layout()
plt.show()

# -------------------- ساخت و رسم Confusion Matrix --------------------
y_preds = []
y_true = []

model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_loader, desc="Making predictions..."):
        X, y = X.to(device), y.to(device)
        y_logits = model_2(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
        y_true.append(y.cpu())

y_pred_tensor = torch.cat(y_preds)
y_true_tensor = torch.cat(y_true)

confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor, target=y_true_tensor)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7),
    show_normed=True
)
plt.show()

#create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create model save
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"  # ← پسوند .pth اضافه شد
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dict
print(f"saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)



# -------------------- پیش‌بینی روی عکس واقعی --------------------
def predict_on_real_image(model, image_path, class_names, device):
    # بارگذاری تصویر
    image = Image.open(image_path).convert("L")  # به خاکستری (grayscale)

    # تغییر اندازه به 28x28 (مثل داده‌های FashionMNIST)
    image = image.resize((28, 28))

    # تبدیل به tensor
    image_tensor = TF.to_tensor(image)

    # نرمال‌سازی (مقادیر بین 0 تا 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # پیش‌بینی
    model.eval()
    with torch.inference_mode():
        pred_logit = model(image_tensor)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_class = torch.argmax(pred_prob, dim=1).item()

    # نمایش نتیجه
    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {class_names[pred_class]}")
    plt.axis(False)
    plt.show()

    return class_names[pred_class]

image_path = "33.jpg"
pred_label = predict_on_real_image(model_2, image_path, class_names, device)
print(f"🧾 مدل حدس زد که این تصویر یک '{pred_label}' است.")



