
import os
import wave
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy import signal
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib

# تنظیمات
DATA_DIR = "E:/Dataset"

TARGET_SR = 16000
TARGET_SECONDS = 3
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS  # نمونه‌ها در 3 ثانیه

N_MFCC = 20         # تغییر به 20
N_FFT = 512
WIN_LEN = 0.025
WIN_STEP = 0.01
N_FILTERS = 40

AUGMENT = True
AUG_PER_SAMPLE = 1
NOISE_LEVEL = 0.005
SHIFT_MAX_SEC = 0.2

#  نام احساسات
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# 1. جمع‌آوری فایل‌ها و لیبل‌ها
# -----------------------------
file_paths = []
labels = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, "unknown")
            else:
                emotion = "unknown"
            file_paths.append(path)
            labels.append(emotion)

print(Counter(labels))
print(f"تعداد کل فایل‌ها: {len(file_paths)}")

# 2. خواندن WAV با resample به 16k
# -----------------------------
def load_wav(path, target_sr=TARGET_SR, target_length=TARGET_LENGTH):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width: {}".format(sampwidth))

        audio = np.frombuffer(raw_data, dtype=dtype)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio.mean(axis=1).astype(dtype)

        audio = audio.astype(np.float32) / (2**(8*sampwidth - 1))

        # resample if needed
        if framerate != target_sr:
            num_samples = int(len(audio) * float(target_sr) / framerate)
            audio = signal.resample(audio, num_samples)

        # fix length
        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start+target_length]
        elif len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')

        return audio, target_sr

# 3. توابع MFCC و دلتا (بدون librosa) + percentiles
# -----------------------------
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10**(mel / 2595.0) - 1.0)

def get_mel_filterbanks(n_filters=40, n_fft=512, sr=16000, low_freq=0, high_freq=None):
    if high_freq is None:
        high_freq = sr / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        if center > left:
            fbanks[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbanks[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbanks

def delta(feat, N=2):
    denom = 2 * sum([i*i for i in range(1, N+1)]) # استاندارد ثابت دلتا
    padded = np.pad(feat, ((N, N), (0,0)), mode='edge')
    delta_feat = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        num = np.zeros((feat.shape[1],))
        for n in range(1, N+1):
            num += n * (padded[t + N + n] - padded[t + N - n]) # فرمول اصلی دلتا
        delta_feat[t] = num / denom
    return delta_feat

def compute_mfcc_features(signal_audio, sr=TARGET_SR,
                          n_mfcc=N_MFCC, n_fft=N_FFT,
                          win_len=WIN_LEN, win_step=WIN_STEP,
                          n_filters=N_FILTERS):
    # pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(signal_audio[0], signal_audio[1:] - pre_emphasis * signal_audio[:-1]) # فرکانس های بالا رو تقویت میکند و فرمول اصلیشه اونم

    # framing
    frame_length = int(round(win_len * sr))
    frame_step = int(round(win_step * sr))
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_length - signal_length,))
    pad_signal = np.append(emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # windowing
    frames *= np.hamming(frame_length)

    # FFT and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n=n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # filter banks
    fbanks = get_mel_filterbanks(n_filters=n_filters, n_fft=n_fft, sr=sr)
    feat = np.dot(pow_frames, fbanks.T)
    feat[feat == 0] = np.finfo(float).eps
    log_feat = np.log(feat)

    # DCT -> MFCC
    mfcc = dct(log_feat, type=2, axis=1, norm='ortho')[:, :n_mfcc]  # (num_frames, n_mfcc)

    # lifter
    cep_lifter = 22
    nframes, ncoeff = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2.) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    # deltas
    mfcc_delta = delta(mfcc, N=2)
    mfcc_delta2 = delta(mfcc_delta, N=2)

    # aggregate: mean, std, 25th, 75th for mfcc, delta, delta2 => 4 stats * 3 groups = 12 stats per coeff
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std  = np.std(mfcc, axis=0)
    mfcc_p25  = np.percentile(mfcc, 25, axis=0)
    mfcc_p75  = np.percentile(mfcc, 75, axis=0)

    d_mean = np.mean(mfcc_delta, axis=0)
    d_std  = np.std(mfcc_delta, axis=0)
    d_p25  = np.percentile(mfcc_delta, 25, axis=0)
    d_p75  = np.percentile(mfcc_delta, 75, axis=0)

    dd_mean = np.mean(mfcc_delta2, axis=0)
    dd_std  = np.std(mfcc_delta2, axis=0)
    dd_p25  = np.percentile(mfcc_delta2, 25, axis=0)
    dd_p75  = np.percentile(mfcc_delta2, 75, axis=0)

    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_p25, mfcc_p75,
        d_mean, d_std, d_p25, d_p75,
        dd_mean, dd_std, dd_p25, dd_p75
    ])  # length = n_mfcc * 12

    return feature_vector

# 4. Augmentation (simple: noise + time shift)
# -----------------------------
#اضفه کردن نویز
def add_noise(audio, noise_level=NOISE_LEVEL):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise
#عقب و جلو بردن صوت
def time_shift(audio, sr=TARGET_SR, max_shift_sec=SHIFT_MAX_SEC):
    max_shift = int(sr * max_shift_sec)
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.concatenate([audio[shift:], np.zeros(shift)])
    elif shift < 0:
        shift = -shift
        return np.concatenate([np.zeros(shift), audio[:-shift]])
    else:
        return audio

# 5. خواندن تمام سیگنال‌ها و (اختیاری) تولید augment
# -----------------------------
processed_data = []
processed_labels = []

for path, label in tqdm(zip(file_paths, labels), desc="Reading WAVs", total=len(file_paths)):
    try:
        audio, sr = load_wav(path, TARGET_SR, TARGET_LENGTH)
        processed_data.append(audio)
        processed_labels.append(label)
        if AUGMENT:
            for _ in range(AUG_PER_SAMPLE):
                aug = audio.copy()
                aug = time_shift(aug, sr=sr, max_shift_sec=SHIFT_MAX_SEC)
                aug = add_noise(aug, noise_level=NOISE_LEVEL)
                processed_data.append(aug)
                processed_labels.append(label)
    except Exception as e:
        print(f"Error loading {path}: {e}")

processed_data = np.array(processed_data)
processed_labels = np.array(processed_labels)

print("تعداد نمونه‌های پردازش شده (شامل augment):", len(processed_data))
print("شکل هر نمونه:", processed_data[0].shape)

# 6. استخراج ویژگی‌ها (MFCC + delta + delta2 aggregated + percentiles)
# -----------------------------
features = []
for audio in tqdm(processed_data, desc="Extracting MFCC features"):
    feat = compute_mfcc_features(audio, sr=TARGET_SR,
                                 n_mfcc=N_MFCC, n_fft=N_FFT,
                                 win_len=WIN_LEN, win_step=WIN_STEP,
                                 n_filters=N_FILTERS)
    features.append(feat)

features = np.array(features)
print("شکل آرایه ویژگی‌ها (aggregate MFCC):", features.shape)  # (N, n_mfcc*12)

# 7. لیبل‌ها و تقسیم دیتاست
# -----------------------------
le = LabelEncoder()
y_all = le.fit_transform(processed_labels)
print("کلاس‌ها:", le.classes_)

# split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    features, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print("تعداد train_full:", X_train_full.shape[0], "تعداد test:", X_test.shape[0])

# 8. نرمال‌سازی (fit روی train_full) و تبدیل به تنسور
# -----------------------------
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_full, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 9. محاسبه class weights و قرار دادن در criterion
# -----------------------------
classes = np.unique(y_train_full)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_full)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 10. مدل MLP بهبود یافته
# -----------------------------
class BetterMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]
output_dim = len(le.classes_)
model = BetterMLP(input_dim, output_dim)

# 11. Optimizer, Scheduler
# -----------------------------
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

# 12. آموزش با tqdm + EarlyStopping + history
# -----------------------------
num_epochs = 100
batch_size = 32
best_val_loss = float("inf")
patience = 15
patience_counter = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0.0
    model.train()

    for i in range(0, X_train_tensor.size()[0], batch_size):
        idx = permutation[i:i+batch_size]
        batch_X = X_train_tensor[idx]
        batch_y = y_train_tensor[idx]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss = epoch_loss / X_train_tensor.size(0)

    # eval
    model.eval()
    with torch.no_grad():
        out_train = model(X_train_tensor)
        _, pred_train = torch.max(out_train, 1)
        train_acc = (pred_train == y_train_tensor).float().mean().item()

        out_val = model(X_test_tensor)
        _, pred_val = torch.max(out_val, 1)
        val_loss = criterion(out_val, y_test_tensor).item()
        val_acc = (pred_val == y_test_tensor).float().mean().item()

    history["train_loss"].append(epoch_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    tqdm.write(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
               f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # scheduler step
    scheduler.step(val_loss)

    # early stopping & save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        tqdm.write("EARLY STOPPING TRIGGERED!")
        break

# 13. ارزیابی نهایی
# -----------------------------
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()

print("Accuracy on test set:", accuracy)

# 14. Confusion matrix, classification report
# -----------------------------
cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test_tensor.numpy(), predicted.numpy(), target_names=le.classes_))

# 15. رسم نمودارها (Loss/Acc + Confusion Matrix)
# -----------------------------
# Loss & Acc curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curve")

plt.subplot(1,2,2)
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy curve")
plt.tight_layout()
plt.show()

# Confusion matrix heatmap (annotated)
plt.figure(figsize=(7,6))
cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)
plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (normalized)")
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
# annotate
thresh = cm_norm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm_norm[i,j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "le.pkl")
