SOFT_TAU = 0.005


def soft_sync(net, target_net):
    for target_param, param in zip(target_net.parameters(),
                                   net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - SOFT_TAU) + param.data * SOFT_TAU)


def hard_sync(net, target_net):
    target_net.load_state_dict(net.state_dict())
