import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# function to initialize distributed process group (1 process per GPU)
# this allow communication among process
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"
    
    # intialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows is not built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows user may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]

        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_in, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_out)

        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    factor = 4
    X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    y_train = y_train.repeat(factor)
    X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False, # false because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)
        # chunk batches across GPUs without overlapping samples:
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False
    )

    return train_loader, test_loader


def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size) # Initilize Process Groups
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_in=2, num_out=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank]) # wrap model with DDP
    # core model is now accessible as model.module
    
    for epoch in range(num_epochs):
        # set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW:
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # NEW: cleanly exit distributed mode


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # rather than running the code as a “regular” Python script (via python ...py) 
    # and manually spawning processes from within Python using multiprocessing.spawn 
    # we will not run this as python script but use torchrun
    # when we run using torchrun, it launces one process per GPU and assign each process a unique rank
    # other metadata as well like world_size, local rank which is read these variables and pass them to main()

    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    # Only print on rank 0 to avoid duplicate prints from each GPU process
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.mps.is_available())
        print("Number of GPUs available:", torch.mps.device_count())

    torch.manual_seed(123) # random seed for reproducibility
    num_epochs = 3
    main(rank, world_size, num_epochs)
    # 1. intialized distributed env via ddp_setup
    # 2. loads training and test dataset
    # 3. Sets up the model, performs training
    # transfer both model and data using .to(rank), where rank correponds to GPU index for current process
    # 4. Wrap the model using DDP, enables synchronized gradient updates across all GPUs
    # 5. Evaluate the model and destroy_process_graph() to properly shut down the process

    # torchrun --nproc_per_node=2 multiple-gpu-train.py
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) multiple-gpu-train.py





    