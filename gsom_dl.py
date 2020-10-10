import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import os

sbn.set()

alpha_ql = 0.2
gamma = 0.9
epsilon = 0.9
alpha_som = 0.01
c_som = 0.00001
k_som = .9
alpha_d = 0.99
delta_som = 0.3
diff_zero = 0.00001
initial_q_values = np.ones((1, 4))*0.00001
actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def init_map():
    # world_map = np.zeros((21, 21))
    # world_map[0] = np.ones(world_map[-1].shape)
    # world_map[-1] = np.ones(world_map[-1].shape)
    # world_map[:, -1] = 1
    # world_map[:, 0] = 1
    # world_map[1, 1] = 2
    world_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                 [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                 [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                 [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                 [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                 [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 3, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    start = np.array([[1, 1]])
    return world_map, start


def step(world_map, actual_pos, action):
    done = False
    reward = 0
    next_pos = actual_pos + actions[action]
    if world_map[next_pos[0][0], next_pos[0][1]] == 1:
        res_pos = actual_pos
    else:
        res_pos = next_pos
        world_map[actual_pos[0][0], actual_pos[0][1]] = 0
        world_map[next_pos[0][0], next_pos[0][1]] = 2
    if world_map[res_pos[0][0],res_pos[0][1]] == 3:
        done = True
        reward = 1
    return world_map, res_pos, reward, done


def som_activation(som, state, diff):
    return np.sum(((state - som)/diff)**2, 1)


def calc_diff(state, next_state, diff):
    actual_diff = np.linalg.norm(next_state - state)
    if actual_diff < diff*delta_som:
        res = diff
    elif diff == diff_zero:
        res = actual_diff
    else:
        res = (1 - alpha_d)*diff + alpha_d*actual_diff
    return res


def get_action(som, state, qtable, diff):
    prototypes = som_activation(som, state, diff)
    best_proto = np.argmin(prototypes)
    q_values = qtable[best_proto]
    if np.random.rand() > epsilon:
        action = np.argmax(q_values)
    else:
        action = np.random.randint(3)
    return action, best_proto


def push_val_in_pos(array, value, pos):
    array_bef = array[:pos]
    array_after = array[pos:]
    return np.concatenate([array_bef, value, array_after])


def add_node(som, qtable, next_state, diff):
    prototypes = som_activation(som, next_state, diff)
    best_proto = np.argmin(prototypes)
    best_proto_dist = np.min(prototypes)
    if best_proto_dist > k_som:
        som = push_val_in_pos(som, next_state, best_proto)
        qtable = push_val_in_pos(qtable, initial_q_values, best_proto)

    return som, qtable, best_proto


def update_som(som, next_state, best_proto):
    r_win = np.ones((som.shape[0], 1))*som[best_proto]
    exp_fact = (r_win - som)**2
    exp_fact = -exp_fact/c_som
    # alpha_i = alpha_som*exp(-(||r_win-r_i||^2)/c)
    alpha_i = alpha_som*np.exp(exp_fact)
    som = (1 - alpha_i)*som + alpha_i*next_state
    return som


def update_q_values(qtable, proto_choosen, action, reward, best_proto):
    actual_value = qtable[proto_choosen][action]
    qtable[proto_choosen][action] = actual_value\
        + alpha_ql * (reward
                      + gamma * np.max(qtable[best_proto])
                      - actual_value)
    return qtable


def main():
    global epsilon
    max_actions = 500
    clear = lambda: os.system('clear')

    som = np.ones((1, 2))
    qtable = np.ones((1, 4))
    qtd_actions = list()
    for epi in range(500):
        world_map, state = init_map()
        done = False
        act_idx = 0
        diff = diff_zero
        while not done:
            action, proto_choosen = get_action(som, state, qtable, diff)
            world_map, next_state, reward, done = step(world_map,
                                                       state,
                                                       action)
            if epi%100==0:
                clear()
                print(world_map)
                import time
                time.sleep(0.1)
            done = done or act_idx > max_actions
            act_idx += 1
            som, qtable, best_proto = add_node(som, qtable, next_state, diff)
            som = update_som(som, next_state, best_proto)
            qtable = update_q_values(qtable, proto_choosen, action,
                                     reward, best_proto)
            diff = calc_diff(state, next_state, diff)
            state = next_state
            epsilon -= 0.01
        qtd_actions.append(act_idx)
    plt.plot(qtd_actions)
    plt.show()


if __name__ == "__main__":
    main()
