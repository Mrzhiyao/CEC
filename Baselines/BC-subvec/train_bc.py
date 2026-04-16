import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import data.dataset as dataset
import time

def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10):
    train_batch_state, train_batch_action, train_batch_reward, train_batch_next_state, train_dones =  train_data
    
    np.random.seed(config.RANDOM_SEED)
    model.to(device)
    best_val_loss = float('inf')
    stop_counter = 0
    best_model = model


    train_loss = 0
    epochs_ran = 0
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        batch_loss = 0
        # for x in len(train_batch_state):
        mini_batch_loss = 0
        
        state_batch = train_batch_state
        correct_actions = train_batch_action.squeeze() 

        optimizer.zero_grad()
        predicted_actions = model(state_batch.to(device))

        cur_loss = criterion(predicted_actions, correct_actions.long().to(device))
        cur_loss.backward()
        optimizer.step()
        mini_batch_loss = cur_loss.item()

        batch_loss += mini_batch_loss
        
        train_loss += batch_loss

        # 损失函数计算
        val_loss = loss(model, device, val_data, criterion)
        print(f'Validation loss: {val_loss} at epoch {epoch + 1}/{config.MAX_EPOCHS}')
        if val_loss < best_val_loss:
            print(f'New best validation loss: {val_loss}\n old best validation loss: {best_val_loss}')
            best_val_loss = val_loss
            stop_counter = 0
            best_model = model
            config.save_model(model, name=f"BC_{epoch + 1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            epochs_ran = epoch + 1
            break
    config.save_model(best_model, name="final_BC")
    # print(f'Final training loss: {train_loss/epochs_ran:.4f} after {epochs_ran} epochs')


def loss(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    test_batch_state, test_batch_action, test_batch_reward, test_batch_next_state, test_dones = val_loader
    with torch.no_grad():
       
        batch_loss = 0
        state_batch = test_batch_state
        correct_actions = test_batch_action.squeeze() 
        predicted_actions = model(state_batch.to(device))
        batch_loss += criterion(predicted_actions, correct_actions.long().to(device)).item()

        # batch_loss /= config.NUM_STEPS_ENV
        val_loss += batch_loss
    val_loss /= len(test_batch_state)
    return val_loss


def test_accuracy(model, device, test_data):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loader = DataLoader(test_data, **config.PARAMS)
    with torch.no_grad():  # No need to track gradients during inference
        for environments, actions in test_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            agent_start_positions = []

            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])
            for step_idx in range(config.NUM_STEPS_ENV):
                # Prepare the test batch
                state_batch = dataset.extract_env_windows(environments, agent_start_positions, config.WINDOW_LEN)
                state_batch = np.asarray(state_batch, dtype=np.int64)
                state_batch = torch.from_numpy(state_batch).float().to(device)
                state_batch = state_batch.unsqueeze(1)  # Ensure the correct shape [batch_size, 1, 60, 5]

                # Get the predicted actions from the model
                predicted_actions = model(state_batch)

                # Assuming predicted_actions is a tensor of probabilities, apply argmax to get the predicted class
                predicted_classes = torch.argmax(predicted_actions, dim=1)
                action_idxs = [x for x, y in agent_start_positions]
                correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]
                # Convert actions to a tensor on the same device
                correct_actions = torch.LongTensor(correct_actions).to(device)
                agent_start_positions = dataset.update_agent_pos(agent_start_positions, expert_paths)
                # Calculate how many predictions are correct
                correct += (predicted_classes == correct_actions).sum().item()
                total += correct_actions.size(0)

    accuracy = correct / total  # Calculate accuracy
    return accuracy



def get_action(state, model, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    state = torch.unsqueeze(state, dim=0)
    predicted_actions =  model(state)
    return predicted_actions