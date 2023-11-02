from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
import torch.functional as F
import torch
from sklearn import metrics

def train_one_epoch(model: torch.nn.modules, dl: DataLoader, optimizer : Optimizer, loss_fn ):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dl):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        last_loss = loss.item()


    return last_loss

def training_loop(model : torch.nn.Module, train_dl : DataLoader, val_dl : DataLoader, epochs = 1, learning_rate = 0.01):
    optimizer = Adam(params=model.parameters(True), lr=learning_rate, betas=[0.9, 0.999])
    loss_fn = CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")

        model.train()
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn)

        model.eval()
        running_vloss = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_dl)
        print(f"train_loss = {train_loss :.4f}, val_loss = {avg_vloss :.4f}")

def evaluate(model : torch.nn.Module, val_dl : DataLoader):
    loss_fn = CrossEntropyLoss()
    model.eval()
    running_vloss = 0
    temperature = 0.01 # Controls softmax preds
    
    # Disable gradient computation and reduce memory consumption.

    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i, vdata in enumerate(val_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            # Use the outputs to get a decision rule via softmax. This is done to get a tangible result for 
            # Accuracy and f1
            probas = torch.softmax(voutputs / temperature, -1)
            pred = torch.multinomial(probas, 1).item()
            gt = vlabels.item()

            predictions.append(pred)
            ground_truths.append(gt)


    avg_vloss = running_vloss / len(val_dl)
    accuracy = metrics.accuracy_score(ground_truths, predictions)
    f1 = metrics.f1_score(ground_truths, predictions, average="weighted")

    print(f"loss = {avg_vloss :.4f}")
    print(f"accuracy = {accuracy :.4f}")
    print(f"f1 = {f1 :.4f}")
    
    return avg_vloss, accuracy, f1

    
