import numpy as np
import sklearn.gaussian_process as gp
import tensorflow as tf
from scipy.stats import norm
from scipy.optimize import minimize


""" 
bayesian optimization of the loss function
"""
MAX_LOSS = 1e8    
def expected_improvement(x, gaussian_process, losses, n_params):
    x_to_predict = x.reshape(-1, n_params)
    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    
    loss_opt = np.min(losses)

    scaling_factor = -1
    
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_opt) / sigma
        expected_improvement = scaling_factor * (mu - loss_opt) * norm.cdf(Z) \
                               + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0
    
    return -1 * expected_improvement

def sample_next_hyperparameter(func, gaussian_process, losses, bounds, n_restarts=100):
    best_x = None
    best_value = MAX_LOSS
    n_params = bounds.shape[0]
    
    for starting_point in np.random.uniform(bounds[:, 0],
                                            bounds[:, 1],
                                            (n_restarts, n_params)):
        res = minimize(fun=func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, losses, n_params))
        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x
                 
    return best_x
                               
def bayesian_optimization(n_iters, sample_loss, bounds, n_initial=5,
                          gp_params=None, alpha=1e-5, epsilon=1e-7):
    x_list = []
    y_list = []
    
    n_params = bounds.shape[0]
    # initialize with random xp and yp
    assert n_initial > 0
    for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial, bounds.shape[0])):
        x_list.append(params)
        print(params)
        y_list.append(sample_loss(params))
    
    
    xp = np.array(x_list)
    yp = np.array(y_list)
    
    with open('xp_yp.txt', 'a') as f:
        f.write('initial random points:\n')
        for i, t in enumerate(zip(xp, yp)):
            x = t[0]
            y = t[1]
            f.write('%s)x : ' % str(i))
            for j in x:
                f.write('%s ' % str(j))
            f.write('y : %s\n' % str(y))
        f.write('iterations:\n')
    
    # creating the gp
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)
    
    print('Started iterations of bayesian model...')
    for i in range(n_iters):
        model.fit(xp, yp)
        
        next_sample = sample_next_hyperparameter(expected_improvement, model, yp, bounds=bounds, n_restarts=100)
        
        # remove duplicates
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
            
        next_loss = sample_loss(next_sample)
    
        x_list.append(next_sample)
        y_list.append(next_loss)
        with open('xp_yp.txt', 'a') as f:
            it = n_initial + i
            f.write('%s)x : ' % str(it))
            for j in x_list[-1]:
                f.write('%s ' % str(j))
            f.write('y : %s\n' % str(y_list[-1]))
            
        xp = np.array(x_list)
        yp = np.array(y_list)
        
    return xp, yp