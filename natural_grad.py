#%%
import jax 
import jax.numpy as np 
import numpy as onp

from functools import partial
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense, Relu

# from tqdm.notebook import tqdm 
from tqdm import tqdm 

from jax.config import config
config.update("jax_debug_nans", True) # break on nans
config.update('jax_enable_x64', True) # use float64 for extra precision? -- cg round off error 

#%%
def make_float64(params):
    return jax.tree_map(lambda x: x.astype(np.float64), params)

net_init, net_apply = stax.serial(
    Dense(128), Relu,
    Dense(10),
)
in_shape = (-1, 784)
rng = jax.random.PRNGKey(0)
out_shape, params = net_init(rng, in_shape)
params = make_float64(params)

#%%
import datasets 
train_images, train_labels, test_images, test_labels = datasets.mnist()
train_images, test_images = train_images.astype(np.float64), test_images.astype(np.float64)
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
    lls = jax.vmap(partial(log_likelihood, params))(batch)
    return lls.mean()

def mean_cross_entropy(params, batch):
    losses = jax.vmap(partial(cross_entropy, params))(batch)
    loss = losses.mean()
    return loss 

def accuracy(params, batch):
    x, y = batch
    probs = net_apply(params, x)
    correct = y[probs.argmax()]
    return correct

@jax.jit
def step(i, opt_state, batch):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)
    return loss, opt_update(i, grads, opt_state), grads

# #%%
# ### sample code to test things out 
# batch_size = 2
# N = len(train_images)
# train_idxs = onp.arange(N) # training idxs 
# onp.random.shuffle(train_idxs) # shuffle training idxs
# batch_idxs = onp.array_split(train_idxs, N//batch_size) # batch em

# for _, idxs in enumerate(batch_idxs): # training loop 
#     x = train_images[idxs]
#     y = train_labels[idxs]
#     break 

# opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
# _, params = net_init(rng, in_shape) # init model
# params = make_float64(params)
# opt_state = opt_init(params) # init optim 
# batch = (x, y)
# x.shape, y.shape 

#%%
def hvp(J, w, v):
    return jax.jvp(jax.grad(J), (w,), (v,))[1]

# z = logits, J = fcn: logits->loss
def gnh_vp(f, J, w, v):
    # J v 
    z, R_z = jax.jvp(f, (w,), (v,))
    # H (J v)
    R_gz = hvp(J, z, R_z)
    # (H (J v))^T J = v^T (J^T H J) 
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(R_gz)[0]

def fisher_vp(f, w, v):
    # J v 
    _, R_z = jax.jvp(f, (w,), (v,))
    # (J v)^T J = v^T (J^T J) 
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(R_z)[0]

@jax.jit
def natural_emp_step(i, opt_state, batch):
    params = get_params(opt_state)

    loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)
    f = lambda w: mean_cross_entropy(w, batch)
    fvp = lambda v: fisher_vp(f, params, v)
    ngrad, _ = jax.scipy.sparse.linalg.cg(fvp, grads, maxiter=10) # approx solve

    return loss, opt_update(i, ngrad, opt_state), ngrad

def sample_label(logits, rng):
    idx = jax.random.categorical(rng, logits, -1)
    one_hot = jax.nn.one_hot(idx, 10)
    return one_hot

def sample_label_rng(x, rng):
    rng, subkey = jax.random.split(rng, 2)
    return rng, sample_label(x, subkey)

def tree_mvp_dampen(mvp, lmbda=0.1):
    dampen_fcn = lambda mvp_, v_: mvp_ + lmbda * v_
    damp_mvp = lambda v: jax.tree_multimap(dampen_fcn, mvp(v), v)
    return damp_mvp

@jax.jit
def natural_step(i, opt_state, batch, rng):
    params = get_params(opt_state)
    
    # sample model labels 
    x, _ = batch
    logits = net_apply(params, x)
    
    new_labels = jax.lax.scan(lambda rng, x: sample_label_rng(x, rng), rng, logits)[1]
    new_labels = jax.lax.stop_gradient(new_labels) # stop backprop on the targets
    sampled_batch = (x, new_labels)

    # gradients on true labels
    loss, grads = jax.value_and_grad(mean_cross_entropy)(params, batch)

    # fisher on sampled labels 
    f = lambda w: mean_cross_entropy(w, sampled_batch)
    fvp = lambda v: fisher_vp(f, params, v)
    fvp = tree_mvp_dampen(fvp, lmbda=1e-8) # dampen (F_w + I * lmbda)^-1 \nabla L
    ngrad, _ = jax.scipy.sparse.linalg.cg(fvp, grads, maxiter=10) # approx solve

    return loss, opt_update(i, ngrad, opt_state), ngrad

# %%
%timeit step(0, opt_state, (x, y)) # 1.61 ms ± 143 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit natural_emp_step(0, opt_state, (x, y)) # 3.15 ms ± 572 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit natural_step(0, opt_state, (x, y), rng) # 3.95 ms ± 227 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

#%%
from torch.utils.tensorboard import SummaryWriter

def train(step_fcn, exp_name, seed=0, rng_in=False):
    global opt_init, opt_update, get_params

    # re-producibility 
    onp.random.seed(seed)

    N = len(train_images)
    train_idxs = onp.arange(N) # training idxs 

    writer = SummaryWriter(comment=exp_name) # metrics 

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

    rng = jax.random.PRNGKey(seed)
    _, params = net_init(rng, in_shape) # init model
    params = make_float64(params)
    opt_state = opt_init(params) # init optim 

    step_i = 0 
    for e in tqdm(range(epochs)):
        train_loss = 0
        onp.random.shuffle(train_idxs) # shuffle training idxs
        batch_idxs = onp.array_split(train_idxs, N//batch_size) # batch em

        for _, idxs in enumerate(tqdm(batch_idxs)): # training loop 
            x = train_images[idxs]
            y = train_labels[idxs]
            
            # optim 
            if rng_in: 
                rng, subkey = jax.random.split(rng)                
                loss, opt_state, grads = step_fcn(step_i, opt_state, (x, y), subkey)
            else:
                loss, opt_state, grads = step_fcn(step_i, opt_state, (x, y))
            
            # log training distrib.
            params = get_params(opt_state)  
            for i, (p, g) in enumerate(zip(params, grads)):
                if p or g: # ignore activations (e.g `()`)
                    (pw, pb), (gw, gb) = p, g
                    try: 
                        writer.add_histogram(f'w{i}/value', onp.asarray(pw), step_i)
                        writer.add_histogram(f'w{i}/grad', onp.asarray(gw), step_i)
                        writer.add_histogram(f'b{i}/value', onp.asarray(pb), step_i)
                        writer.add_histogram(f'b{i}/grad', onp.asarray(gb), step_i)
                    except: # errors out when values are too large
                        pass 

            writer.add_scalar('train/loss', loss.item(), step_i)

            train_loss += loss 
            step_i += 1 

        params = get_params(opt_state)    
        test_acc = jax.jit(jax.vmap(accuracy, (None, 0), 0))(params, (test_images, test_labels)).mean()
        print(f'epoch: {e} test_acc: {test_acc * 100:.2f} train_loss: {train_loss:.2f}')
        writer.add_scalar('test/accuracy', test_acc.item(), e)

        train_loss = 0 

    params = get_params(opt_state)    
    return params 

# %%
epochs = 2
batch_size = 128
lr = 1e-3

# #%%
# params = train(step, 'vanilla_SGD_f64')

# #%%
# params = train(natural_emp_step, 'natural(emp)_f64')

# %%
params = train(natural_step , 'natural_f64', rng_in=True)

# %%
