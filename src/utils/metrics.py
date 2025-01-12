"""
This dictionary returns the loss function used for training, indexed by the dataset name
returns a tuple with the function + a string to print
"""
import torch
from sklearn.metrics import f1_score, average_precision_score

train_losses_dict = {
    #TUDataset:
    "MUTAG": (),
    "ENZYMES": (torch.nn.CrossEntropyLoss(), "CE"),
    "PROTEINS": (torch.nn.CrossEntropyLoss(), "CE"),
    "COLLAB": (),
    "IMDB": (),
    "REDDIT": (),    

    #GNNBenchmark:
    "CIFAR10": (),
    "MNIST": (),
    "CLUSTER": (),
    "PATTERN": (),
    "TSP" : (),

    #LRGBDataset:
    "COCO-SP": (torch.nn.CrossEntropyLoss(), "CE"),
    "PascalVOC-SP" : (torch.nn.CrossEntropyLoss(), "CE"),
    "COCO-SP-mini": (torch.nn.CrossEntropyLoss(), "CE"),
    "PascalVOC-SP-mini" : (torch.nn.CrossEntropyLoss(), "CE"),
    "PCQM-Contact": (),
    "Peptides-func": (torch.nn.BCELoss(), "BCE"),
    "Peptides-struct": (torch.nn.MSELoss(), "MSE")
}

"""
This dictionary returns the metric we use to report our model performance
"""
report_metric_dict = {
    #TUDataset:
    "MUTAG": (),
    "ENZYMES": ("Acc"),
    "PROTEINS": ("Acc"),
    "COLLAB": (),
    "IMDB": (),
    "REDDIT": (),    

    #GNNBenchmark:
    "CIFAR10": (),
    "MNIST": (),
    "CLUSTER": (),
    "PATTERN": (),
    "TSP" : (),

    #LRGBDataset:
    "COCO-SP": ("F1"),
    "PascalVOC-SP" : ("F1"),
    "COCO-SP-mini": ("F1"),
    "PascalVOC-SP-mini" : ("F1"),
    "PCQM-Contact": (),
    "Peptides-func": ("AP"),
    "Peptides-struct": ("MAE")
}

def evaluate_acc(model, val_loader, args, atom_encoder=None, bond_encoder=None):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    loss_fn, loss_fn_name = train_losses_dict[args.data]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Print first batch predictions
            if batch_idx == 0:  # Only for first batch
                #print("\nFirst 20 samples of validation batch:")
                #print(f"Labels:      {batch.y[:20].cpu().numpy()}")
                
                batch.to(args.device)
                if atom_encoder is not None:
                    batch = atom_encoder(batch)
                if bond_encoder is not None:
                    batch = bond_encoder(batch)
                edge_attr = getattr(batch, 'edge_attr', None)
                output = model(batch.x, batch.edge_index, batch.batch,
                             edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), rwse=(None if not hasattr(batch, "random_walk_pe") else batch.random_walk_pe))
                
                predictions = output.argmax(dim=-1)
                #print(f"Predictions: {predictions[:20].cpu().numpy()}")
                #print(f"Raw outputs:\n{output[:20].cpu().detach().numpy()}\n")
            
            batch.to(args.device)
            if atom_encoder is not None:
                batch = atom_encoder(batch)
            if bond_encoder is not None:
                batch = bond_encoder(batch)
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), full_batch=batch )
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
            if loss_fn_name == "BCE":
                output = torch.sigmoid(output)
            loss = loss_fn(output, batch.y)
            total_loss += loss.item()

            predictions = output.argmax(dim=-1)
            correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy


def evaluate_f1(model, val_loader, args, atom_encoder=None, bond_encoder=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    loss_fn, loss_fn_name = train_losses_dict[args.data]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Print first batch predictions
            if batch_idx == 0:  # Only for first batch
                # print("\nFirst 20 samples of validation batch:")
                # print(f"Labels:      {batch.y[:20].cpu().numpy()}")
                
                batch.to(args.device)
                if atom_encoder is not None:
                    batch = atom_encoder(batch)
                if bond_encoder is not None:
                    batch = bond_encoder(batch)
                edge_attr = getattr(batch, 'edge_attr', None)
                output = model(batch.x, batch.edge_index, batch.batch,
                             edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), rwse=(None if not hasattr(batch, "random_walk_pe") else batch.random_walk_pe))
                
                predictions = output.argmax(dim=-1)
                # print(f"Predictions: {predictions[:20].cpu().numpy()}")
                # print(f"Raw outputs:\n{output[:20].cpu().detach().numpy()}\n")
            
            batch.to(args.device)
            if atom_encoder is not None:
                batch = atom_encoder(batch)
            if bond_encoder is not None:
                batch = bond_encoder(batch)
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), full_batch=batch)
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            loss = loss_fn(output, batch.y)
            total_loss += loss.item()

            predictions = output.argmax(dim=-1)
            all_preds.append(predictions.cpu())
            all_targets.append(batch.y.cpu())

    # Combine all predictions and targets into a single array
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute macro-weighted F1 score
    f1 = f1_score(all_targets, all_preds, average='macro')

    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, f1

def evaluate_ap(model, val_loader, args, atom_encoder=None, bond_encoder=None):
    """
    Evaluate the model on the validation set using Average Precision (AP) metric.
    
    Args:
        model: The model to evaluate.
        val_loader: Validation data loader.
        args: Arguments including device and loss function.
    
    Returns:
        avg_loss: Average loss over the validation set.
        avg_ap: Average Precision (AP) score for the multi-label classification task.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    loss_fn, loss_fn_name = train_losses_dict[args.data]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch.to(args.device)
            if atom_encoder is not None:
                batch = atom_encoder(batch)
            if bond_encoder is not None:
                batch = bond_encoder(batch)
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                           edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), rwse=(None if not hasattr(batch, "random_walk_pe") else batch.random_walk_pe))
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            # Compute loss
            if loss_fn_name == "BCE":
                output = torch.sigmoid(output)
            loss = loss_fn(output, batch.y.float())
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(output.cpu())
            all_targets.append(batch.y.cpu())

    # Combine all predictions and targets into single tensors
    all_preds = torch.cat(all_preds).numpy()  # Shape: [num_samples, num_classes]
    all_targets = torch.cat(all_targets).numpy()  # Shape: [num_samples, num_classes]

    # Compute Average Precision (AP) for each class and take the mean
    avg_ap = average_precision_score(all_targets, all_preds, average='macro')

    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, avg_ap

def evaluate_mae(model, val_loader, args, atom_encoder=None, bond_encoder=None):
    """
    Evaluate the model on the validation set using Mean Absolute Error (MAE).
    
    Args:
        model: The model to evaluate.
        val_loader: Validation data loader.
        args: Arguments including device and other configurations.
        atom_encoder: Optional AtomEncoder for processing node features.

    Returns:
        avg_loss: Average loss over the validation set.
        mae: Mean Absolute Error (MAE) for regression.
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    loss_fn = torch.nn.MSELoss()  # Use MSE as the training loss

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch.to(args.device)

            # Apply atom_encoder if provided
            if atom_encoder is not None:
                batch = atom_encoder(batch)
            if bond_encoder is not None:
                batch = bond_encoder(batch)
            # Forward pass
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                           edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), rwse=(None if not hasattr(batch, "random_walk_pe") else batch.random_walk_pe))
            
            if output.dim() == 1:
                output = output.unsqueeze(0)

            # Compute MSE loss
            loss = loss_fn(output, batch.y.float())  # Ensure targets are float for regression
            total_loss += loss.item()

            # Compute MAE for evaluation
            mae = torch.abs(output - batch.y.float()).mean().item()
            total_mae += mae * batch.y.size(0)  # Weighted sum
            total_samples += batch.y.size(0)

    avg_loss = total_loss / len(val_loader)
    mae = total_mae / total_samples

    return avg_loss, mae