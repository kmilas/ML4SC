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

# def create_IO_data(u,y,na,nb):
#     X = []
#     Y = []
#     for k in range(max(na,nb), len(y)):
#         X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
#         Y.append(y[k])
#     return np.array(X), np.array(Y)

# na = 11
# nb = 10
# X, Y = create_IO_data(u_train, th_train, na, nb)

# # Split into train/val/test
# Xtemp, Xtest_final, Ytemp, Ytest_final = train_test_split(X, Y, test_size=0.15, random_state=42)
# Xtrain_final, Xval, Ytrain_final, Yval = train_test_split(Xtemp, Ytemp, test_size=0.1765, random_state=42)
# Xtrain_final = np.expand_dims(Xtrain_final, axis=-1)
# Xval = np.expand_dims(Xval, axis=-1)

# np.save("Xtrain_final.npy", Xtrain_final)
# np.save("Ytrain_final.npy", Ytrain_final)
# np.save("Xval.npy", Xval)
# np.save("Yval.npy", Yval)
# np.save("Xtest_final.npy", Xtest_final)
# np.save("Ytest_final.npy", Ytest_final)