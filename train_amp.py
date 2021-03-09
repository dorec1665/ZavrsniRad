from pathlib import Path
from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
import time
import gc
import model
from torch.utils.data import random_split
import os

epochs = 20
batch_size = 50
start_time = None

def get_model_size(model):
    torch.save(model.state_dict(), 'my_model_amp.pt')
    size = os.path.getsize('my_model_amp.pt')/1e6
    os.remove('my_model_amp.pt')
    return size

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer(message):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + message)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

def evaluate(name, eval_data, eval_model, eval_loss):
    with torch.no_grad():
        N = len(eval_data) * batch_size
        num_batches = N // batch_size
        corrent_cnt = 0
        avg_loss = 0
        for batch_eval_idx, batch_eval_data in enumerate(eval_data):
            eval_inputs, eval_labels = batch_eval_data
            eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)

            with torch.cuda.amp.autocast():
                eval_outputs = eval_model(eval_inputs)

            predicted_labels = torch.argmax(eval_outputs, dim=1)
            corrent_cnt += (eval_labels == predicted_labels).sum().item()
            avg_loss += eval_loss(eval_outputs, eval_labels).item()

        accuracy = corrent_cnt / N * 100
        avg_loss = avg_loss / num_batches

        print(name + "accuracy = %.2f" % accuracy)
        print(name + "avg loss = %.2f" % avg_loss)


DATA_DIR = Path(__file__).parent / 'dataset' / 'CIFAR10'

transform = transforms.Compose([
    transforms.ToTensor()
])

train_val_data = CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
test_data = CIFAR10(DATA_DIR, train=False, transform=transform)

val_size = 5000
train_size = len(train_val_data) - val_size

train_data, val_data = random_split(train_val_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

model = model.resnet18()
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

criterion = torch.nn.CrossEntropyLoss()

n_batch = len(train_data) // batch_size

scaler = torch.cuda.amp.GradScaler()

start_timer()
for epoch in range(epochs):
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print("epoch: {}, step: {}/{}, batch_loss: {}"
                  .format(epoch, batch_idx, n_batch, loss.item()))

    evaluate("Train: ", train_loader, model, criterion)
    evaluate("Validation: ", val_loader, model, criterion)
    scheduler.step()

end_timer("Amp: ")
evaluate("Test: ", test_loader, model, criterion)
print("Model size: {}".format(get_model_size(model)) + "MB")

