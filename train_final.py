from math import log
import torch
from networks import CatecoricalAC, ModularCatecoricalAC, StateEncoder
import gym
import numpy as np
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import mpi_avg_grads, setup_pytorch_for_mpi, sync_params

from spinup.utils.mpi_tools import mpi_avg, mpi_fork, num_procs, proc_id
import torchvision.transforms as T
from buffer import Buffer
import spinup.algos.pytorch.vpg.core as core
import time
from torch.optim.adam import Adam
from environment import MyEnv
from pprint import pprint
from tqdm import tqdm

PROCESSES = 5
MODULAR = False
BATCH_SIZE = 32
AUTOENCODER_EPOCHS = 40


pi_losses = []
v_losses = []
autoenc_losses = []
avg_returns = []
state_list = []

def train(
    actor_critic=CatecoricalAC, 
    ac_kwargs=dict(model_state_path="experiments/ppo_good/vars.pkl"),
    algo="vpg", 
    grayscale_transform=True,
    seed=0,
    env_interactions_per_process=1000,
    gas_factor=0.3,
    break_factor=1, 
    epochs=100,
    action_skip=4, 
    image_stack=4,
    gamma=0.9, 
    store_states=False,
    clip_ratio=0.2,
    lr=3e-4, 
    pi_lr=1e-3,
    vf_lr=1e-3,
    train_pi_iters=20,
    train_v_iters=20,
    lam=0.9,
    target_kl=0.05,
    logger_kwargs=dict(output_dir=f"experiments/{time.time()}"), 
    save_freq=10):


    m = "_modular" if MODULAR else ""
    output_dir=f"experiments/{epochs}_epoch{m}_{algo}{time.time()}"
    logger_kwargs.update(output_dir=output_dir)
    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MyEnv(stack_images=image_stack, skip_actions=action_skip, grayscale=grayscale_transform, gas_factor=gas_factor, break_factor=break_factor)
    h, w, c = env.observation_space
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    sync_params(ac)

      # Count variables
    var_counts = core.count_vars(ac)
    logger.log('\nNumber of parameters: %d\n'%var_counts)

    # Set up experience buffer
    buf = Buffer((c, h, w), 1, env_interactions_per_process, gamma, lam)

    def compute_loss_vpg(data):
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data["ret"]

        pi, v, logp = ac(obs, act)
        # Policy loss
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        # Value loss
        loss_v = ((v - ret) ** 2).mean()
        return loss_pi, pi_info, loss_v
    
    def compute_loss_ppo(data):
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data["ret"]

        pi, v, logp = ac(obs, act)
        # Policy loss
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # Value loss
        loss_v = ((v - ret) ** 2).mean()
        return loss_pi, pi_info, loss_v

    optimizer = Adam(ac.parameters(), lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)


    logger.setup_pytorch_saver(ac)
    
    def update():
        data = buf.get()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info, loss_v_old = compute_loss_vpg(data)
        loss_pi.backward()
        # for param in ac.pi.parameters():
        #     param.grad.data.clamp_(-1, 1)
        mpi_avg_grads(ac.pi) # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            _, _, loss_v = compute_loss_vpg(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Get loss and info values after update
        pi_l_new, pi_info_new, v_l_new = compute_loss_vpg(data)
        pi_l_new = pi_l_new.item()
        v_l_new = v_l_new.item()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info['ent']
        logger.store(LossPi=loss_pi.item(), LossV=loss_v.item(), KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - loss_pi.item()), DeltaLossV=(v_l_new - loss_v_old.item()))

    def update_vpg_combined():
        data = buf.get()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info, loss_v_old = compute_loss_vpg(data)
        loss_pi.backward(retain_graph=True)
        loss_v_old.backward()
        mpi_avg_grads(ac) # average grads across MPI processes
        optimizer.step()

        # Get loss and info values after update
        pi_l_new, pi_info_new, v_l_new = compute_loss_vpg(data)
        pi_l_new = pi_l_new.item()
        v_l_new = v_l_new.item()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info['ent']
        logger.store(LossPi=loss_pi.item(), LossV=loss_v_old.item(), KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - loss_pi.item()), DeltaLossV=(v_l_new - loss_v_old.item()))


    def update_ppo():
        data = buf.get()

        pi_l_old, pi_info_old, v_l_old = compute_loss_ppo(data)
        pi_l_old = pi_l_old.item()
        v_l_old = v_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info, _ = compute_loss_ppo(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            _, _, loss_v = compute_loss_ppo(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        
    def update_ppo_combined():
        data = buf.get()

        pi_l_old, pi_info_old, v_l_old = compute_loss_ppo(data)
        pi_l_old = pi_l_old.item()
        v_l_old = v_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            optimizer.zero_grad()
            loss_pi, pi_info, loss_v = compute_loss_ppo(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward(retain_graph=True)
            loss_v.backward()
            mpi_avg_grads(ac)    # average grads across MPI processes
            optimizer.step()

        logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    

    for epoch in range(epochs):
        ep_return, ep_len = 0, 0
        state = env.reset()
        for t in range(env_interactions_per_process):
            a, v, logp = ac.step(state)
            next_state, r, done = env.step(a)
            ep_return += r
            ep_len += 1
            
            if store_states:
                state_list.append(state)

            buf.store(state, a, r, v, logp)
            logger.store(VVals=v)
            
            state = next_state

            epoch_ended = t == env_interactions_per_process - 1

            if done or epoch_ended:
                if epoch_ended and not done:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = ac.step(state)
                else:
                    v = 0
                buf.finish_path(v)
                if done:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_return, EpLen=ep_len)
                ep_return, ep_len = 0, 0
                state = env.reset()
                done = False

        if algo == "vpg":
            update()
            # update_vpg_combined()
        else:
            update_ppo()
            # update_ppo_combined()

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state(ac.state_dict(), epoch)

        

        if proc_id() == 0:
            stats_dict = logger.epoch_dict
            pi_losses.append(stats_dict["LossPi"])
            v_losses.append(stats_dict["LossV"])
            avg_returns.append(stats_dict["EpRet"])


        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*env_interactions_per_process * PROCESSES)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)

        

        logger.dump_tabular()

        
    
    env.close()

def train_autoencoder(states, output_dir):
    # create batches 
    env = MyEnv()
    ac = ModularCatecoricalAC(env.observation_space, env.action_space)
    ac.to("cuda")
    batches = []
    optimizer = Adam(ac.autoencoder_parameters(), lr=0.0001)
    for i in range(0, len(states), BATCH_SIZE):
        batches.append(states[i: i + BATCH_SIZE])
    total = len(states)
    batch_len = len(batches)
    desc = f"Training on {total} states in {batch_len} batches for {AUTOENCODER_EPOCHS} epochs"
    for epoch in tqdm(range(AUTOENCODER_EPOCHS), desc=desc):
        for batch in batches:
            tensor_batch = torch.cat(batch)
            tensor_batch = tensor_batch.to("cuda")
            optimizer.zero_grad()
            out = ac.forward_autoenc(tensor_batch)
            loss = ((out - tensor_batch) ** 2).mean()
            loss.backward()
            optimizer.step()
        print(loss.item())
    ac.to("cpu")
    torch.save(ac.encoder.state_dict(), f"{output_dir}/encoder_weights.pt")
    torch.save(ac.decoder.state_dict(), f"{output_dir}/decoder_weights.pt")

            

def main():
    if MODULAR:
        # output_dir=f"experiments/100_epoch_modular_ppo_combined{time.time()}"
        # output_dir="experiments/1639331332.810533"
        encoder_dir = "experiments/encoder_pretrain_no_relu_66_img"
        # logger_kwargs = dict(output_dir=output_dir)
        mpi_fork(PROCESSES)
        ac_kwargs = dict(encoder_path=f"{encoder_dir}/encoder_weights.pt")
        train(epochs=300, actor_critic=ModularCatecoricalAC, ac_kwargs=ac_kwargs)
    else:
        # output_dir=f"experiments/100_epoch_ppo{time.time()}"
        # # output_dir="experiments/test"
        # logger_kwargs = dict(output_dir=output_dir)
        mpi_fork(PROCESSES)
        # ac_kwargs = dict(model_state_path="experiments/100_epoch_ppo1639491161.8668485/vars80.pkl")
        ac_kwargs={}
        train(ac_kwargs=ac_kwargs)

def train_encoder():
    output_dir=f"experiments/{time.time()}"
    output_dir=f"experiments/encoder_pretrain_66_img_v2"
    logger_kwargs = dict(output_dir=output_dir, output_fname="pretrain.txt")
    # ac_kwargs = dict(model_state_path="eexperiments/ppo_modular_good/vars100.pkl")
    ac_path = "experiments/ppo_modular_good/pyt_save/model100.pt"
    with open(ac_path, "rb") as f:
        ac = torch.load(f)
    env = MyEnv(gas_factor=0.3)
    states = []
    s = env.reset()
    for i in tqdm(range(30000)):
        a, _, _ = ac.step(s)
        s, r, d = env.step(a)
        states.append(s)
        if d:
            s = env.reset()
    # train(epochs=1, env_interactions_per_process=10000,ac_kwargs={}, store_states=True, train_v_iters=1, train_pi_iters=1, logger_kwargs=logger_kwargs)
    train_autoencoder(states, output_dir)

if __name__ == "__main__":
    main()
    # train_encoder()