import numpy as np
# from sklearn.model_selection import train_test_split

out = np.load('training-val-test-data.npz')
th_train = out['th']
u_train = out['u']

# Estimate omega using finite differences
dt = 0.025
omega = np.gradient(th_train, dt)

# Save for training the NOE model
np.save("theta.npy", th_train)
np.save("omega.npy", omega)
np.save("u.npy", u_train)