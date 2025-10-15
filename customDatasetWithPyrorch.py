import torch
from matplotlib.pyplot import title
from torch import nn
import requests
import zipfile
import pathlib
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms , datasets
from torch.utils.data import DataLoader , Dataset
from typing import Tuple, List,Dict
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import pandas as pd
import  torchvision
#----------------------------------------------
# تنظیم دستگاه اجرا (CPU)
device = torch.device("cpu")
#----------------------------------------------

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
zip_path = data_path / "pizza_steak_sushi.zip"

#----------------------------------------------
# بررسی وجود دیتاست
if image_path.is_dir() and any(image_path.iterdir()):
    print(f"{image_path} directory already exists and is not empty.")
else:
    print(f"{image_path} directory does not exist or is empty. Downloading dataset...")

    # اطمینان از وجود پوشه data
    data_path.mkdir(parents=True, exist_ok=True)

    # دانلود دیتاست
    response = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza_steak_sushi.zip ...")
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print(f"File size: {len(response.content)/1024/1024:.2f} MB")

    # از حالت فشرده خارج کردن فایل
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping pizza_steak_sushi.zip ...")
        zip_ref.extractall(data_path)

#----------------------------------------------
# تابع نمایش ساختار پوشه‌ها
def walk_through(dir_path):
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'")

#----------------------------------------------
# نمایش ساختار دیتاست
walk_through(image_path)
#----------------------------------------------
# مشخص کردن مسیر آموزش و تست
train_dir=image_path / "train"
test_dir=image_path / "test"
# print(train_dir)
# print(test_dir)
#----------------------------------------------
# انتخاب تصادفی تصویر از دیتاست

image_path_list=list(image_path.glob("*/*/*.jpg"))   # گرفتن مسیر تمام تصاویر
random_image_path=random.choice(image_path_list)     # انتخاب یک تصویر تصادفی
image_class=random_image_path.parent.name            # نام کلاس از نام پوشه گرفته می‌شود
img=Image.open(random_image_path)                    # باز کردن تصویر

# نمایش اطلاعات تصویر
# print(f" random image path: {random_image_path}")
# print(f" image class: {image_class}")
# print(f"image height: {img.height}")
# print(f"image width: {img.width}")
# print(img)

#---------------------------------------------------
# تبدیل تصویر به آرایه
img_as_array=np.asarray(img)

# نمایش تصویر با matplotlib
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"image class: {image_class}, image shape {img_as_array.shape} -> [height , width , color_channels]")
# plt.axis(False)
# plt.show()
# plt.close('all')
#----------------------------------------------------
# اعمال تبدیل‌ها (Transforms) روی تصاویر
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),         # تغییر اندازه تصویر
    transforms.RandomHorizontalFlip(p=0.5),   # برعکس کردن تصادفی افقی تصویر
    transforms.ToTensor()                     # تبدیل به Tensor
])
#-----------------------------------------------------
# تابع نمایش تصاویر قبل و بعد از Transform
def plot_transformed_images(image_paths:list , transfrom,n=3,seed=None):
    if seed:
        random.seed(seed)
    random_image_paths=random.sample(image_paths,k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig,ax=plt.subplots(nrows=1,ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"original \nsize: {f.size}")
            ax[0].axis(False)

            transformed_image=transfrom(f).permute(1,2,0) # تغییر ترتیب برای نمایش در matplotlib
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"transformed \nshape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"class: {image_path.parent.stem}",fontsize=16)

plot_transformed_images(image_paths=image_path_list,
                        transfrom=data_transform,
                        n=3,
                        seed=42)
# plt.show()
# plt.close('all')
#----------------------------------------------------------
# ساخت دیتاست آموزش و تست با ImageFolder
train_data=datasets.ImageFolder(root=train_dir,transform=data_transform,target_transform=None)
test_data=datasets.ImageFolder(root=test_dir,transform=data_transform)

# گرفتن نام کلاس‌ها به‌صورت لیست
class_names=train_data.classes

# گرفتن نام کلاس‌ها به‌صورت دیکشنری
class_dict=train_data.class_to_idx

# دریافت یک نمونه از دیتاست برای بررسی
img , label=train_data[0][0], train_data[0][1]
# print(f"image tensor: \n{img}")
# print(f"imagg shape: {img.shape}")
# print(f"label: {label}")
# print(f"label datatype: {type(label)}")

# تغییر ترتیب ابعاد برای نمایش
img_permute=img.permute(1,2,0)

# نمایش تصویر و اطلاعات آن
# print(f"img shape: {img.shape}")
# print(f"img_permute shape: {img_permute.shape}")

# plt.figure(figsize=(10, 7))
# plt.imshow(img_permute)
# plt.axis("off")
# plt.title(class_names[label],fontsize=14)
# plt.show()
# plt.close('all')

#------------------------------------------------------
# 🔹 از اینجا به بعد باید داخل شرط main باشد تا در ویندوز خطا ندهد
if __name__ == "__main__":
    # ساخت DataLoader برای دیتاست آموزش و تست
    BATCH_SIZE=32
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,num_workers=0, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,num_workers=0, shuffle=False)

    img , label=next(iter(train_dataloader))

    # نمایش شکل داده‌ها
    # print(f"img shape: {img.shape}")
    # print(f"label.shape: {label.shape}")

    # نمایش اطلاعات کلاس‌ها
    # print(train_data.class_to_idx , train_data.classes)

    # مسیر پوشه هدف (train)
    target_dictionary=train_dir
    # print(f"target_dictionary: {target_dictionary}")

    # استخراج نام کلاس‌ها از پوشه‌ها
    class_names_found=sorted([entry.name for entry in list(os.scandir(target_dictionary))])
    # print(f"class_names_found: {class_names_found}")

def find_classes(directory:str)-> Tuple[List[str],Dict[str, int]]:
#1.get the class names by scanning the target dictionary
    classes=sorted(entry.name for entry in os.scandir(directory)if entry.is_dir())
#2.raise an error if class names could not be found
    if not classes:
         raise FileNotFoundError(f"couldn't find any classes in {directory}")
#3.create a dictionary of index labels (computers prefer numbers rather than strings as labels)
    class_to_idx={class_names:i for i,class_names in enumerate(classes)}
    return classes, class_to_idx

# print(find_classes(target_dictionary))
#-----------------------------------------
class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        """
        سازنده کلاس دیتاست
        ------------------
        targ_dir : مسیر پوشه اصلی داده‌ها (مثلاً data/train)
        transform : عملیات پیش‌پردازش (مثل Resize, ToTensor و غیره)
        """
        # گرفتن مسیر تمام فایل‌های jpg داخل زیرپوشه‌ها
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform

        # استفاده از تابع find_classes برای گرفتن کلاس‌ها و ایندکس آن‌ها
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        """
        باز کردن تصویر با اندیس داده شده
        """
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")  # اطمینان از RGB بودن تصویر

    def __len__(self) -> int:
        """
        طول دیتاست (تعداد کل تصاویر)
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        گرفتن یک داده (تصویر و برچسب)
        خروجی: (image_tensor, label)
        """
        # لود تصویر از مسیر
        img = self.load_image(index)

        # استخراج نام کلاس از نام پوشه (مثلاً 'pizza', 'steak', ...)
        class_name = self.paths[index].parent.name

        # تبدیل نام کلاس به عدد با استفاده از دیکشنری کلاس‌ها
        label = self.class_to_idx[class_name]

        # اعمال Transformها در صورت وجود
        if self.transform:
            img = self.transform(img)

        # برگرداندن تصویر و برچسب عددی
        return img,label

# img , label =train_data[0]
# print(img,label)
#-----------------------------------------------------------
#create a transforms
train_transform=transforms.Compose([transforms.Resize(size=(64, 64)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),])

test_transform=transforms.Compose([transforms.Resize(size=(64, 64)),
                                   transforms.ToTensor()])
#test out ImageFolderCustom
train_data_custom=ImageFolderCustom(targ_dir=train_dir, transform=train_transform)
test_data_custom=ImageFolderCustom(targ_dir=test_dir, transform=test_transform)
# print(train_data_custom)
# print(test_data_custom)
#------------------------------------------------------------
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    """
    تابع نمایش چند تصویر تصادفی از یک Dataset
    ----------------------------------------
    dataset : دیتاست ورودی (مثلاً ImageFolderCustom یا ImageFolder)
    classes : لیست نام کلاس‌ها
    n : تعداد تصاویری که باید نمایش داده شوند (حداکثر 10)
    display_shape : آیا شکل تصویر (ابعاد) هم نشان داده شود؟
    seed : عدد ثابت برای تکرارپذیری انتخاب تصادفی
    """

    # جلوگیری از انتخاب بیش از 10 تصویر
    if n > 10:
        print("⚠️ خطا: مقدار n نباید بیشتر از 10 باشد.")
        n = 10
        display_shape = False

    # اگر seed داده شده، مقدار تصادفی را ثابت می‌کند
    if seed is not None:
        random.seed(seed)

    # انتخاب n اندیس تصادفی از دیتاست
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # آماده‌سازی شکل نمایش
    plt.figure(figsize=(16, 8))

    # حلقه برای نمایش تصاویر انتخاب‌شده
    # for i, sample_idx in enumerate(random_samples_idx):
    #     targ_image, targ_label = dataset[sample_idx]
    #
    #     # تغییر ترتیب ابعاد از [C, H, W] به [H, W, C] برای نمایش با matplotlib
    #     targ_image_adjust = targ_image.permute(1, 2, 0)
    #
    #     # ترسیم تصویر
    #     plt.subplot(1, n, i + 1)
    #     plt.imshow(targ_image_adjust)
    #     plt.axis("off")
    #
    #     # عنوان تصویر (کلاس + ابعاد در صورت نیاز)
    #     if classes:
    #         title = f"class: {classes[targ_label]}"
    #         if display_shape:
    #             title += f"\nshape: {tuple(targ_image_adjust.shape)}"
    #     else:
    #         title = f"label: {targ_label}"
    #
    #     plt.title(title, fontsize=10)
    #
    # plt.show()
    # plt.close('all')

#display random images from the imagefolder crated dataset
display_random_images(train_data,n=5,classes=class_names,seed=None)
#-------------------------------------------------------------------------
#create custom loaded images into DataLoaders
BATCH_SIZE=32
NUM_WORKERS=0
train_dataloader_custom=DataLoader(dataset=train_data_custom,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS, shuffle=True)
test_dataloader_custom=DataLoader(dataset=test_data_custom,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS, shuffle=False)

#get image and label from custom dataloader
img_custom,label_custom=next(iter(train_dataloader_custom))
# print(img_custom.shape,label_custom.shape)
#--------------------------------------------------------------------------
#data augmentation
train_transform=transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                    transforms.ToTensor(),])

test_transform=transforms.Compose([transforms.Resize(size=(224, 224)),
                                   transforms.ToTensor()])

#get all image paths
image_path_list=list(image_path.glob("*/*/*.jpg"))

#plot random transformed images
# plot_transformed_images(
#     image_paths=image_path_list,
#     transfrom=train_transform,
#     n=3,
#     seed=None,
# )
# plt.show()
# plt.close('all')
#----------------------------------------------------------
#MODEL_0 WITHOUT DATA AUGMENTATION
#create simple transform
simple_transform=transforms.Compose([transforms.Resize(size=(64,64)),
                                     transforms.ToTensor()])

#1.load and transform data
train_data_simple=datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple=datasets.ImageFolder(root=test_dir, transform=simple_transform)

#2.turn the datasets into DataLoadres
#setup bath size and number of works
BATCH_SIZE=32
NUM_WORKERS=0

#create dataloaders
train_dataloader_simple=DataLoader(dataset=train_data_simple,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS, shuffle=True)
test_dataloader_simple=DataLoader(dataset=test_data_simple,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS, shuffle=False)
#_______________________________________________________________
#CREATE TINIVGG MODEL CLASS
class TiniVGG(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int)->None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 14 * 14, out_features=output_shape)
        )

    def forward(self, x):
        x=self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.classifier(x)
        #print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x)))

torch.manual_seed(42)
model_0= TiniVGG(input_shape=(3),hidden_units=10,output_shape=len(class_names)).to(device)
# print(model_0)
#-----------------------------------------------------------------------------------------------------------------------
#get a single image batch
image_batch,label_batch=next(iter(train_dataloader_custom))
# print(image_batch.shape,label_batch.shape)
#--------------------------------------------------------------------------------
#try forward pass
# print(model_0(image_batch.to(device)))
#----------------------------------------------------------------------------
# if __name__ == "__main__":
#     s = summary(model_0, input_size=[1, 3, 64, 64])
#-----------------------------------------------------------------------------
#CREATE TRAIN AND TEST LOOPS FUNCTIONS

#create train step
def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:device):
    #put the model in train mode
    model.train()

    #setup train loss and train accuracy values
    train_loss , train_acc = 0, 0

    #loop trough data loader data bathes
    for batch,(X,y) in enumerate(dataloader):
        #send data to the target device
        X,y=X.to(device),y.to(device)

        #1.forward pass
        y_pred=model(X)

        #2.calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #3.optimazer zero gred
        optimizer.zero_grad()

        #4.loos a backward
        loss.backward()

        #5.optimazer step
        optimizer.step()

        #calculate accuracy metric
        y_pred_class=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    #adjust metrics to get average loss and accuracy per batch
    train_loss=train_loss/ len(dataloader)
    train_acc=train_acc/len(dataloader)
    return train_loss, train_acc
#----------------------------------------------------------------------
#create a test step
def test_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:device):
    #put model in eval mode
    model.eval()

    #setup  test loss and test accuracy values
    test_loss, test_acc = 0, 0

    #turn on inference mode
    with torch.inference_mode():
        #loop trough dataloader batchs
        for batch,(X,y) in enumerate(dataloader):
            #send data to the target device
            X,y=X.to(device),y.to(device)

            #1.forward pass
            test_pred_logits=model(X)

            #2.calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            #calculate the accuracy
            test_pred_labels=torch.argmax(test_pred_logits, dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

        #adjust metrics to get average loss and accuracy per batch
        test_loss=test_loss/len(dataloader)
        test_acc=test_acc/len(dataloader)
        return test_loss, test_acc
#------------------------------------------------------------------------------------------
#1.create a train function that takes in various model parameters + optimazer + loss
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,  # 👈 اضافه شد
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: str = "cpu"):

    # 1️⃣ دیکشنری برای ذخیره نتایج
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 2️⃣ حلقه‌ی آموزش
    for epoch in tqdm(range(epochs)):
        # مرحله آموزش
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        # مرحله تست
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # میانگین‌گیری و چاپ نتایج در هر epoch
        print(f"Epoch: {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")

        # ذخیره در دیکشنری نتایج
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
#-------------------------------------------------------
#TRAIN AND EVALUATE MODEL 0
#set random seeds
torch.manual_seed(42)

#set number of epochs
NUM_EPOCHS = 5

#recreate aninstance of TiniVGG
model_0=TiniVGG(input_shape=3,hidden_units=10,output_shape=len(train_data.classes)).to(device)

#setup loss function and optimazer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)

#start the timer
start_time=timer()

#train midel 0
model_0_results=train(model=model_0,train_dataloader=train_dataloader_simple,test_dataloader=test_dataloader_simple,optimizer=optimizer,loss_fn=loss_fn,epochs=NUM_EPOCHS)

#end the timer and print out how long it look
end_time=timer()
print(f"Training time: {end_time-start_time}seconds")
#---------------------------------------------------------
#PLOT THE LOSS CURVES OF MODEL_0
def plot_loss_curves(results: dict[str, List[float]]):
    #get the loss values of the results dictionary (training and test)
    loss=results["train_loss"]
    test_loss=results["test_loss"]

    #get the accuracy values of the results dictionary (training and test)
    acc=results["train_acc"]
    test_accuracy=results["test_acc"]

    #figure out how many epochs there were
    epochs=range(len(results["train_loss"]))

    #setup a plot
    plt.figure(figsize=(15,7))

    #plot rhe loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="Train_Loss")
    plt.plot(epochs, test_loss, label="Test_Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    #plot the accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, label="Train_Accuracy")
    plt.plot(epochs, test_accuracy, label="Test_Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.legend()
plot_loss_curves(model_0_results)
#-------------------------------------------------------------------
#MODEL_1 WITH DATA AUGMENTATION
#create training transform with TrivialAugment
train_transform_trivial=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
])
test_transform_simple=transforms.Compose([transforms.Resize((64,64)),
                                           transforms.ToTensor(),
])
#---------------------------------------------------------------------
#CREATE TRAIN AND TEST dataset AND dataloader WITH DATA AUGMENTATION
#turn image folders into dataset
train_data_augmented=datasets.ImageFolder(root=train_dir,transform=train_transform_trivial)
test_data=datasets.ImageFolder(root=test_dir,transform=test_transform_simple)
#---------------------------------------------------------------------
#turn our Datasets into dataloaders
BATCH_SIZE=32
NUM_WORKERS=0

torch.manual_seed(42)
train_dataloader_augmented=DataLoader(dataset=train_data_augmented,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=True)
test_dataloader_simple=DataLoader(dataset=test_data_simple,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=False)
#------------------------------------------------------------------------
#CONSTRUCT AND TRAIN MODEL 1
#create model_1 and send it do the target device
torch.manual_seed(42)
model_1=TiniVGG(input_shape=3,hidden_units=10,output_shape=len(train_data_augmented.classes)).to(device)
#-------------------------------------------------------------------------
#set a random seeds
torch.manual_seed(42)

#set the number of epochs
NUM_EPOCHS=5

#Setup loss function
loss_fn=nn.CrossEntropyLoss()
optimazer=torch.optim.Adam(params=model_1.parameters(),lr=0.001)

#start the timer
start_time=timer()

#train model 1
model_1_results=train(model=model_1,
                      train_dataloader=train_dataloader_augmented,
                      test_dataloader=test_dataloader_simple,
                      optimizer=optimazer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device="cpu")
#end the timer and print out how long it took
end_time=timer()
print(f"Training time: {end_time-start_time}seconds")
#-----------------------------------------------------------------------
#PLOT THE LOSS CURVES OF MODEL1
plot_loss_curves(model_1_results)
#-----------------------------------------------------------------------
model_0_df=pd.DataFrame.from_dict(model_0_results,orient='index')
model_1_df=pd.DataFrame.from_dict(model_1_results,orient='index')
print(f"model_0_df:\n{model_0_df},\nmodel_1_df:\n{model_1_df}")

#setup a plot
plt.figure(figsize=(15,7))

#get nuber of epochs

# ------------------------ مقایسه‌ی مدل‌ها ------------------------
import matplotlib.pyplot as plt

# ساخت محدوده‌ی epoch بر اساس تعداد ستون‌های دیتا
epochs = range(len(model_0_df.columns))

# ------------------------ نمودار خطای آموزش ------------------------
plt.figure(figsize=(14, 6))

# رسم خطا (loss)
plt.subplot(1, 2, 1)
plt.plot(epochs, model_0_df.loc["train_loss"], label="Model_0 Train Loss", marker='o')
plt.plot(epochs, model_0_df.loc["test_loss"], label="Model_0 Test Loss", marker='o')
plt.plot(epochs, model_1_df.loc["train_loss"], label="Model_1 Train Loss", linestyle='--', marker='x')
plt.plot(epochs, model_1_df.loc["test_loss"], label="Model_1 Test Loss", linestyle='--', marker='x')
plt.title("Loss Comparison Between Models")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# ------------------------ نمودار دقت آموزش ------------------------
plt.subplot(1, 2, 2)
plt.plot(epochs, model_0_df.loc["train_acc"], label="Model_0 Train Accuracy", marker='o')
plt.plot(epochs, model_0_df.loc["test_acc"], label="Model_0 Test Accuracy", marker='o')
plt.plot(epochs, model_1_df.loc["train_acc"], label="Model_1 Train Accuracy", linestyle='--', marker='x')
plt.plot(epochs, model_1_df.loc["test_acc"], label="Model_1 Test Accuracy", linestyle='--', marker='x')
plt.title("Accuracy Comparison Between Models")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#--------------------------------------------------------------------
#MAKING A PREDICTION ON A CUSTOM IMAGE
#download custom image

#setup custom image path
custom_image_path=data_path/"x.jpg"
#--------------------------------------------------------------------
#read im custom image
custom_image_uint8=torchvision.io.read_image(str(custom_image_path))
print(custom_image_uint8)
plt.imshow(custom_image_uint8.permute(1,2,0))
plt.show()
#----------------------------------------------------------------------
#MAKING A PREDICTION ON A CUSTOM IMAGE WITH A TRAINED PYTORCH MODEL
#try to make a prediction on a image in uint8 format

#load in custom image and convert to torch.float32
custom_image=torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
#----------------------------------------------------------------
#creatr transform pipline to resize image
custom_image_transform=transforms.Compose([transforms.Resize((64,64)),])

#transform target image
custom_image_transformed=custom_image_transform(custom_image)
print(f"original shape{custom_image.shape}")
print(f"transformed shape{custom_image_transformed.shape}")
#------------------------------------------------------------------

