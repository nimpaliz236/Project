import torch
from torch import nn
from torchvision import datasets
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm

# -----------------------------------------------------------
# print(torchvision.__version__)
# -----------------------------------------------------------
device = torch.device("cpu")  # فقط CPU
# -----------------------------------------------------------
train_data = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor(), target_transform=None)

class_names = train_data.classes
# print(class_names)

class_to_idx = train_data.class_to_idx
# print(class_to_idx)

image, label = train_data[0]
# plt.imshow(image.squeeze())
# plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4

for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    # plt.show()

BATCH_SIZE = 32
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

# let is check out what what we have created
# print(f"DataLoaders:{train_loader,test_loader}")
# print(f"length of train_loader: {len(train_loader)}")
# print(f"length of test_loader: {len(test_loader)}")

# check out what is inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_loader))
# print(f"train_features_batch: {train_features_batch.shape}")
# print(f"train_labels_batch: {train_labels_batch.shape}")

# show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
# print(f"image size: {img.shape}")
# print(f"label: {label},label size: {label.shape}")

# 3. Model 0: build a basline model

# create a flatten layer
flatten_model = nn.Flatten()

# get a single sample
x = train_features_batch[0]

# flatten the sample
output = flatten_model(x)  # perform forward pass


# print out what happend
# print(f"shape before flattening: {x.shape}")
# print(f"shape after flattening: {output.shape}")

class FashionMNISTmodelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


torch.manual_seed(42)

# setup model with input parameters
model_0 = FashionMNISTmodelV0(input_shape=784, hidden_units=10, output_shape=len(class_names)).to("cpu")

dummy_x = torch.rand([1, 1, 28, 28])
# print(model_0(dummy_x))

# setup loss function and optimazer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)


def print_train_time(start: float, end: float, device: torch.device = None):
    total = end - start
    print(f"train time on {device}: {total:.3f} seconds")
    return total


start_time = timer()
end_time = timer()
a = print_train_time(start=start_time, end=end_time, device="cpu")
print(a)

# set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3

# create training and test loop
for epoch in range(epochs):
    print(f"epoch{epoch}\n------")

    # training
    train_loss = 0
    for batch, (X, Y) in enumerate(train_loader):
        model_0.train()

        # forward pass
        y_pred = model_0(X)

        # calculate loss
        loss = loss_fn(y_pred, Y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"looked at {batch * len(X)} / {len(train_loader.dataset)} samples")

    train_loss = train_loss / len(train_loader)

    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.no_grad():  # Use torch.no_grad() instead of torch.inference_mode()
        for x_test, y_test in test_loader:
            test_pred = model_0(x_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"\ntrain loss:{train_loss:.4f}, test loss:{test_loss:.4f}, test accuracy:{test_acc:.4f}")
    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu,
                                                device=str(next(model_0.parameters()).device))

    torch.manual_seed(42)


    def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
                   accuracy_fn,device=device):
        "returns a dictionary containing the results of model predicting on data_loader"
        loss, acc = 0, 0
        model.eval()
        with torch.no_grad():  # Correct use of 'torch.no_grad()'
            for X, Y in data_loader:
                #make our data device agnostic
                X, Y = X.to(device), Y.to(device)
                # make predictions
                y_pred = model(X)

                # accumulate the loss and accuracy values per batch
                loss += loss_fn(y_pred, Y)
                acc += accuracy_fn(y_true=Y, y_pred=y_pred.argmax(dim=1))

            # scale loss and accuracy to find the average loss/accuracy per batch
            loss /= len(data_loader)
            acc /= len(data_loader)

        return {"model_name": model.__class__.__name__,
                "model_loss": loss.item(), "model_accuracy": acc}


    model_0_results = eval_model(model=model_0, data_loader=test_loader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    print(model_0_results)

#------------------------------------------------------------------------
    # create a model with non-linear and linear layers
    class FashionMNISTmodelV1(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=input_shape, out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=output_shape),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor):
            return self.layer_stack(x)

# Create an instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTmodelV1(input_shape=784, hidden_units=10, output_shape=len(class_names)).to("cpu")

# setup a loss, optimizer and evaluation metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)


def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    train_loss, train_acc = 0, 0

    # put model into training mode
    model.train()

    for batch, (X, Y) in enumerate(data_loader):
        # put data on target device
        X, Y = X.to(device), Y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate loss
        loss = loss_fn(y_pred, Y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=Y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    print(f"train loss:{train_loss:.4f}, train accuracy:{train_acc:.4f}")


def eval_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.no_grad():  # Correct use of 'torch.no_grad()'
        for x_test, y_test in data_loader:
            X, Y = x_test.to(device), y_test.to(device)
            test_pred=model(X)
            test_loss += loss_fn(test_pred, Y)
            test_acc += accuracy_fn(y_true=Y, y_pred=test_pred.argmax(dim=1))

        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)
        print(f"test loss:{test_loss:4f}, test accuracy:{test_acc:.4f}")

        torch.manual_seed(42)
        train_time_end_on_cpu=timer()
        epochs = 3
        for epoch in range(epochs):
            print(f"epoch{epoch}\n------")
            train_step(model=model_1, data_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer,accuracy=accuracy_fn, device=device)
            eval_step(model=model_1, data_loader=test_loader, loss_fn=loss_fn, accuracy=accuracy_fn, device=device)
        train_time_end_on_cpu=timer()
        total_train_time_model_1=print_train_time(start=train_time_end_on_cpu, end=train_time_end_on_cpu,device=device)
        print(total_train_time_model_1)

model_1_results=eval_model(model=model_1, data_loader=test_loader, loss_fn=loss_fn, accuracy_fn=accuracy_fn,device=device)
print(model_1_results)

#-----------------------------------------------------------------------------------------------------------------------------

#create a convolutional neural network
class FashionMNISTmodelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block1=nn.Sequential(nn.Conv2d(in_channels=input_shape,
                                                 out_channels=hidden_units,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block2=nn.Sequential(nn.Conv2d(in_channels=hidden_units,
                                                 out_channels=hidden_units,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                         nn.ReLU(),

                         nn.Conv2d(in_channels=hidden_units,
                                   out_channels=hidden_units,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=hidden_units,
                                                  out_features=output_shape))
        def forward(self, x):
            x=self.conv_block1(x)
            print(x.shape)
            x=self.conv_block2(x)
            print(x.shape)
            x=self.classifier(x)
            return x

torch.manual_seed(42)
model_2=FashionMNISTmodelV2(input_shape=1,
                            hidden_units=10,
                            output_shape=len(class_names)).to("cpu")

torch.manual_seed(42)
images=torch.randn(size=(32, 3, 64, 64))
print(f"images.shape:{images.shape}")
print(f"images.shape:{test_image.shape}")