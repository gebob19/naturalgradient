#%%
import jax 
import scipy

import numpy as onp
import jax.numpy as np 

from jax.flatten_util import ravel_pytree
from functools import partial
from jax import random, vmap, jit, grad
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense, Relu

# from tqdm.notebook import tqdm 
from tqdm import tqdm 

from jax.config import config
config.update("jax_debug_nans", True) # break on nans
# config.update('jax_enable_x64', True) # use float64 for extra precision?

#%%
def make_float64(params):
    return jax.tree_map(lambda x: x.astype(np.float64), params)

net_init, net_apply = stax.serial(
    Dense(128), Relu,
    Dense(10), 
)
in_shape = (-1, 784)
rng = random.PRNGKey(0)
out_shape, params = net_init(rng, in_shape)
# params = make_float64(params)

#%%
import datasets 
train_images, train_labels, test_images, test_labels = datasets.mnist()
# train_images, test_images = train_images.astype(np.float64), test_images.astype(np.float64)
train_images.shape, test_labels.shape

#%%
def log_likelihood(params, batch):
    x, y_oh = batch
    logits = net_apply(params, x)
    probs = jax.nn.softmax(logits, 0)
    ll = np.log((probs * y_oh).sum(0))
    return ll 

def cross_entropy(params, batch):
    loss = -log_likelihood(params, batch)
    return loss

def mean_log_likelihood(params, batch):
    lls = vmap(partial(log_likelihood, params), 0, 0)(batch)
    return lls.mean()

def mean_cross_entropy(params, batch):
    losses = vmap(partial(cross_entropy, params), 0, 0)(batch)
    loss = losses.mean()
    return loss 

def accuracy(params, batch):
    x, y = batch
    probs = net_apply(params, x)
    correct = y[probs.argmax()]
    return correct

@jit
def step(i, opt_state, batch):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)
    return loss, opt_update(i, grads, opt_state), grads

def grad2fisher(g):
    if len(g.shape) == 1: # bias edge case 
        g = g[:, np.newaxis]
    return g @ g.T

@jit
def natural_step_emp(i, opt_state, batch):
    params = get_params(opt_state)
    # gradient for loss
    loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)
    fisher = jax.tree_map(grad2fisher, grads)

    # approx solve for gradient 
    cg_solve = lambda f, g: jax.scipy.sparse.linalg.cg(f, g, maxiter=100)[0]
    natural_grad_emp = jax.tree_multimap(cg_solve, fisher, grads)

    return loss, opt_update(i, natural_grad_emp, opt_state), natural_grad_emp

def natural_fisher(params, batch):
    # compute fisher 
    ll_grads = jax.grad(mean_log_likelihood)(params, batch) 
    fisher = jax.tree_map(grad2fisher, ll_grads) 
    return fisher 

def natural_step(n_samples): ## lil trick to take n_samples as input (otherwise can't @jit it)

    @jit
    def natural_step_inner(i, opt_state, batch, rng):
        params = get_params(opt_state)
        # gradient for loss
        loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)
        
        # sample from model predictions
        x, _ = batch 
        logits = net_apply(params, x)
        rng, *subkeys = random.split(rng, 1 + n_samples) # jax random things 
        sample_idxs = jax.tree_map(lambda k: jax.random.categorical(k, logits, -1), subkeys)
        y_oh = vmap(lambda i: jax.nn.one_hot(i, 10), 0, 0)(sample_idxs)
        y_oh = y_oh.transpose(1, 0, 2) # (nsamples, batch, n_classes)
        y_oh = jax.lax.stop_gradient(y_oh) 
        
        # compute fisher 
        fishers = vmap(lambda y: natural_fisher(params, (x, y)))(y_oh)
        fisher = jax.tree_map(lambda f: f.mean(0), fishers)

        # approx solve for gradient 
        cg_solve = lambda f, g: jax.scipy.sparse.linalg.cg(f, g, maxiter=100)[0]
        natural_grad = jax.tree_multimap(cg_solve, fisher, grads)

        # gradient clipping (exploding gradients problem :?)
        natural_grad = jax.tree_map(lambda f: np.clip(f, a_min=-10, a_max=10), natural_grad)

        return loss, opt_update(i, natural_grad, opt_state), natural_grad, rng
    
    return natural_step_inner

#%%
from torch.utils.tensorboard import SummaryWriter

def train(step_fcn, exp_name, seed=0, rng_in_out=False):
    global opt_init, opt_update, get_params

    # re-producibility 
    onp.random.seed(seed)

    N = len(train_images)
    train_idxs = onp.arange(N) # training idxs 

    writer = SummaryWriter(comment=exp_name) # metrics 

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    rng = random.PRNGKey(seed)
    _, params = net_init(rng, in_shape) # init model
    # params = make_float64(params)
    opt_state = opt_init(params) # init optim 

    step_i = 0 
    for e in tqdm(range(epochs)):
        train_loss = 0
        onp.random.shuffle(train_idxs) # shuffle training idxs
        batch_idxs = onp.array_split(train_idxs, N//batch_size) # batch em

        for idxs in tqdm(batch_idxs): # training loop 
            x = train_images[idxs]
            y = train_labels[idxs]
            
            # optim 
            if rng_in_out: 
                loss, opt_state, grads, rng = step_fcn(step_i, opt_state, (x, y), rng)
            else:
                loss, opt_state, grads = step_fcn(step_i, opt_state, (x, y))
            
            # log training distrib.
            params = get_params(opt_state)  
            for i, (p, g) in enumerate(zip(params, grads)):
                if p or g: # ignore activations (e.g `()`)
                    (pw, pb), (gw, gb) = p, g
                    writer.add_histogram(f'w{i}/value', onp.asarray(pw), step_i)
                    writer.add_histogram(f'w{i}/grad', onp.asarray(gw), step_i)
                    writer.add_histogram(f'b{i}/value', onp.asarray(pb), step_i)
                    writer.add_histogram(f'b{i}/grad', onp.asarray(gb), step_i)

            writer.add_scalar('train/loss', loss.item(), step_i)

            train_loss += loss 
            step_i += 1 

        params = get_params(opt_state)    
        test_acc = jit(vmap(accuracy, (None, 0), 0))(params, (test_images, test_labels)).mean()
        writer.add_scalar('test/accuracy', test_acc.item(), e)

        print(f'epoch: {e} test_acc: {test_acc * 100:.2f} train_loss: {train_loss:.2f}')
        train_loss = 0 

    params = get_params(opt_state)    
    return params 

# %%
epochs = 2
batch_size = 128
lr = 1e-3

#%%
params = train(step, 'vanilla SGD')

#%%
params = train(natural_step_emp, 'natural(emp)')

# %%
params = train(natural_step(n_samples=1) , 'natural(1sample)', rng_in_out=True)

# %%
params = train(natural_step(n_samples=5), 'natural(5sample)', rng_in_out=True)

# %%
