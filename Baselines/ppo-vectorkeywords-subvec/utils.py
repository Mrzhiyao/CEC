import torch
import pickle
import random
import os
_SHARED_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'expert_demonstrations')

def save(args, save_name, model, wandb, ep):
    import os
    save_dir = './trained_models/'
    os.chmod(save_dir, 0o777)  # 给文件所有用户完全权限
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

def collect_random_main_train_test(dataset):

    with open(os.path.join(_SHARED_DATA, 'memory_sim_new_reward_high_cost.pkl'), 'rb') as file:
        loaded_lists_sim = pickle.load(file)

    with open(os.path.join(_SHARED_DATA, 'memory_relu_new_reward.pkl'), 'rb') as file:
        loaded_lists_relu = pickle.load(file)

    with open(os.path.join(_SHARED_DATA, 'memory_sim_subvec_reward_high_cost.pkl'), 'rb') as file:
        loaded_lists_sim_subvec = pickle.load(file)

    with open(os.path.join(_SHARED_DATA, 'memory_relu_subvec_reward.pkl'), 'rb') as file:
        loaded_lists_relu_subvec = pickle.load(file)

    with open(os.path.join(_SHARED_DATA, 'memory_rl_subvec_reward.pkl'), 'rb') as file:
        loaded_lists_rl_subvec = pickle.load(file)

    experience = []
    for _ in range(1000):
        state = loaded_lists_sim[_+500][0]
        action = loaded_lists_sim[_+500][1]
        reward = loaded_lists_sim[_+500][2]
        next_state = loaded_lists_sim[_+500][3]
        done = 0

        add_state = state
        add_next = next_state
        add_state_part1 = add_state[0:2]
        add_state_part2 = add_state[3:5]
        add_state_part3 = add_state[6:8]
        add_state_part4 = add_state[9:]

        result_state = add_state_part1 + add_state_part2 + add_state_part3 + add_state_part4

        add_next_part1 = add_next[0:2]
        add_next_part2 = add_next[3:5]
        add_next_part3 = add_next[6:8]
        add_next_part4 = add_next[9:]

        result_next = add_next_part1 + add_next_part2 + add_next_part3 + add_next_part4

        result_state_add = result_state[0:15] + result_state[15:19] + result_state[35:39] + result_state[55:59]
        result_next_add = result_next[0:15] + result_next[15:19] + result_next[35:39] + result_next[55:59]

        experience.append([result_state_add, action, reward, result_next_add, done])

    for _ in range(len(loaded_lists_sim_subvec)):
        state = loaded_lists_relu[_][0]
        action = loaded_lists_relu[_][1]
        reward = loaded_lists_relu[_][2]
        next_state = loaded_lists_relu[_][3]
        done = 0

        add_state = state
        add_next = next_state
        add_state_part1 = add_state[0:2]
        add_state_part2 = add_state[3:5]
        add_state_part3 = add_state[6:8]
        add_state_part4 = add_state[9:]

        result_state = add_state_part1 + add_state_part2 + add_state_part3 + add_state_part4

        add_next_part1 = add_next[0:2]
        add_next_part2 = add_next[3:5]
        add_next_part3 = add_next[6:8]
        add_next_part4 = add_next[9:]

        result_next = add_next_part1 + add_next_part2 + add_next_part3 + add_next_part4
        result_state_add = result_state[0:15] + result_state[15:19] + result_state[35:39] + result_state[55:59]
        result_next_add = result_next[0:15] + result_next[15:19] + result_next[35:39] + result_next[55:59]
        experience.append([result_state_add, action, reward, result_next_add, done])

    for _ in range(0, len(experience)):
        dataset.add(experience[_][0], experience[_][1], experience[_][2], experience[_][3], experience[_][4])

