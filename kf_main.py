import numpy as np
from numpy.linalg import pinv
import pylab
import os

base_path = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_kf/useful/'
link_base_path = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_kf/sources/selected_time_period/selected_link_npy'

save_path_2_step = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_kf/results/2_step/'
save_path_3_step = '/home/xi/Documents/Research/OD_prediction/DL_prediction/data/new_kf/results/3_step/'

files = sorted(os.listdir(link_base_path))

# The general structure of Kalman filter
def kf_predict(X, P, F, Q):
	X = np.dot(F, X)
	
	P = np.dot(F, np.dot(P, F.T)) + Q

	return X, P


def kf_update(X, P, Y, H, Num):
	IM = np.dot(H, X)

	#print(X.shape, P.shape, Y.shape, H.shape)

	S = np.dot(H, np.dot(P, H.T)) + 0.0000001*np.identity(50)
	#print(S,H,P)
	K = np.dot(P, np.dot(H.T, pinv(S)))
	X = X + np.dot(K, (Y-IM))
	P = P - np.dot(K, np.dot(H, P))
	return X, P



# transition equation
F = np.load(base_path+'FQ_npy/F.npy')
Q = np.load(base_path+'FQ_npy/Q.npy')

#print('Q Value: ',np.sum(Q))

#print(Q)




# begin estimation **************************************************

# initial value
for day in range(30):


	print('Current Day: ', day)

	N_iter = 10

	P = np.load(base_path+'init_npy/P_0.npy')

	for iter_i in range(N_iter):

		print('Current Iteration: ', iter_i)

		X = np.load(base_path+'init_npy/X_0.npy').reshape(650*9,1)

		Num = 13

		for i in range(Num):

			print('index: ', i)
			#print(files[day*14+i])
		
			Y = np.load(base_path+'Y_npy/'+files[day*14+i])

			A = np.load(base_path+'ass_ratio_npy/'+files[day*14+i]).reshape(50,650*9)  # have a test before using

				#X_real = np.load(base_path+'od_npy/'+'06_00_'+str(25+i)+'.npy')
		
			X, P = kf_predict(X, P, F, Q)

			new_X_2, new_P_2 = kf_predict(X, P, F, Q)
			new_X_3, new_P_3 = kf_predict(new_X_2, P, F, Q)

			if iter_i == N_iter-1:
				#np.save(save_path+files[day*14+i], new_X)
				np.save(save_path_2_step+files[day*14 + i + 1], new_X_2)

				if i < 12: 
					np.save(save_path_3_step+files[day*14 + i + 2], new_X_3)

			X, P = kf_update(X, P, Y, A, Num)

			print('Current P: ', np.sum(P))
