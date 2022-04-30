# v0
# v1 -> Added dynamic model and dataset import

#%% Imports
import os
import random
import datetime
import importlib
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils.utils as utils

#%% Function definitions
def run_epoch_f(args, mode, model, criterion, optimizer,
                logger, data_loader, device, epoch):
   
    if mode == 'Train':
        model.train()
        mode_desc = 'Training_epoch'
    elif mode == 'Validation':
        model.eval()
        mode_desc = 'Validating_epoch'
    elif mode == 'Test':
        model.eval()
        mode_desc = 'Testing_epoch'

    total_entries = 0
    sum_loss = 0
    Y_pred = []
    Y_gr_truth = []
    
    for step_idx, X in tqdm(enumerate(data_loader),
                              total = len(data_loader), 
                              desc = mode_desc):
        
        # Move data to cuda
        if next(model.parameters()).is_cuda:
            X = X.to(device)
        
        # Train step
        if mode == 'Train':
            # Zero gradients
            optimizer.zero_grad()
            #Forward + backward + optimize
            pred_score = model(X)
            # Compute loss
            loss = criterion(pred_score, X)
            # Backpropagate
            loss.backward()
            # Update model
            optimizer.step()
        
        # Eval / Test step        
        else:
            with torch.no_grad(): 
                pred_score = model(X)
                # Compute loss
                loss = criterion(pred_score, X)
        
        # Book-keeping
        current_batch_size = X.size()[0]
        total_entries += current_batch_size
        sum_loss += (loss.item() * current_batch_size)

        #Append predictions to lists
        #if mode == 'Test':
        #    Y_pred += pred_score.cpu().detach().numpy().tolist()
        #    Y_gr_truth += X.cpu().detach().numpy().tolist()
       
        # Log train step
        if mode == 'Train':
            logger.info(f'Epoch {epoch + 1} of {args.n_epochs}' +
                        f' Step {step_idx + 1:,} of {len(data_loader):,}')
        
    # Compute metrics
    avg_loss = sum_loss / total_entries
    
    # Print & log results
    print(f'\n{mode} loss: {avg_loss:.4f}')
    logger.info(f'{mode} loss: {avg_loss:.4f}')
    
    return avg_loss, Y_pred, Y_gr_truth

#%% Main
def main():
    # Arg parsing
    args = utils.parse_args_f()
    
    # Path initialization train-dev
    path_train_dataset = os.path.join(args.input_dir, 'model_train.pt')
    path_dev_dataset = os.path.join(args.input_dir, 'model_dev.pt')
    path_test_dataset = os.path.join(args.input_dir, args.test_file)
    
    # Create ouput dir if not existing
    utils.make_dir_f(args.output_dir)
    
    # Instantiate logger
    logger = utils.get_logger_f(args.output_dir)
      
    # Global and seed initialization
    random.seed = args.seed
    _ = torch.manual_seed(args.seed)

    # Import model and dataclass
    model_module_name = args.model_filename.split('.')[0]
    model_module = importlib.import_module(model_module_name)
                                       
    # Generate dataloaders
    if args.task == 'Train':
        # Load datasets
        print(f'{datetime.datetime.now()}: Loading data')
        train_dataset = torch.load(path_train_dataset)
        dev_dataset = torch.load(path_dev_dataset)
        print(f'{datetime.datetime.now()}: Done')
        # Convert to toy data if required
        if eval(args.train_toy_data) == True:
            train_dataset = train_dataset[0:args.len_train_toy_data]
            dev_dataset = dev_dataset[0:args.len_train_toy_data]
        # Instantiate dataclasses
        train_dataset = model_module.VAE_dataset(train_dataset)
        dev_dataset = model_module.VAE_dataset(dev_dataset)
        # Instantiate dataloaders
        train_dl = DataLoader(train_dataset,
                              batch_size = args.batch_size_train,
                              shuffle = eval(args.shuffle_train),
                              drop_last = eval(args.drop_last_train))
        dev_dl = DataLoader(dev_dataset,
                            batch_size = int(args.batch_size_train * args.dev_train_ratio),
                            shuffle = False)

    elif args.task == 'Test':
        # Load datasets
        print(f'{datetime.datetime.now()}: Loading data')
        test_dataset = torch.load(path_test_dataset)
        print(f'{datetime.datetime.now()}: Done')
        # Instantiate dataclasses
        test_dataset = model_module.VAE_dataset(test_dataset)
        # Instantiate dataloaders
        test_dl = DataLoader(test_dataset,
                             batch_size = int(args.batch_size_test),
                             shuffle = False)
       
    # Instantiate model
    model = model_module.VAE_model(args)

    # Set device and move model to device
    model, device = utils.model_2_device_f(args, model)

    # Instantiate optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr,
                                 weight_decay = args.wd)
    criterion = nn.MSELoss()

    # Train procedure
    if args.task == 'Train':
        # Save model parameters    
        utils.save_args_f(args)
        
        # Initializaztion
        train_loss_history = []
        val_loss_history = []
        start_time = datetime.datetime.now()
        
        for epoch in tqdm(range(args.n_epochs), desc = 'Training dataset'):        
            # Train
            mode = 'Train'    
            train_loss, _, _ = run_epoch_f(args, mode, model, criterion,
                                           optimizer, logger, train_dl,
                                           device, epoch)
            train_loss_history.append(train_loss)
    
            # Validate
            mode = 'Validation'
            val_loss, _, _ = run_epoch_f(args, mode, model, criterion,
                                         optimizer, logger, dev_dl,
                                         device, epoch)
            val_loss_history.append(val_loss)
    
            # Save checkpoint
            if eval(args.save_model_steps) == True and epoch >= args.save_step_cliff:
                utils.save_checkpoint_f(args, epoch, model, optimizer, train_loss)
        
        # Save model
        if eval(args.save_final_model) == True:
            utils.save_checkpoint_f(args, epoch, model, optimizer, train_loss)
        
        # Save train results
        utils.save_train_results_f(args, train_loss_history, val_loss_history, start_time)

    # Test procedure
    elif args.task == 'Test':
        mode = 'Test'
        # Load model
        path_model = os.path.join(args.output_dir, args.model_file)
        model.load_state_dict(torch.load(path_model)['model_state_dict'])
        # Compute predictions
        avg_loss, Y_pred, Y_gr_truth = run_epoch_f(args, mode, model,
                                                   criterion, optimizer,
                                                   logger, test_dl,
                                                   device, epoch = None)
        # Save test results
        utils.save_test_results_f(args, avg_loss, Y_pred, Y_gr_truth)

if __name__ == "__main__":
    main()
