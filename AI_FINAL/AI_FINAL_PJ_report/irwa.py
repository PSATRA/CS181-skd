import numpy as np
def QP_solver(A1,A2,b1,b2,g,H):
    def W(x, epsi, weight):
        w_list = []
        for i in range(len(b1)):
            val = A1[i] @ x + b1[i]
            w_i = 1.0 / np.sqrt(val**2 + epsi[i]**2)
            w_list.append(w_i * weight)

        for j in range(len(b2)):
            val = max(A2[j] @ x + b2[j], 0)
            w_j = 1.0 / np.sqrt(val**2 + epsi[len(b1) + j]**2)
            w_list.append(w_j * weight)
        return np.array(w_list)
    def V(x):
        v_list = []
        for i in range(len(b1)):
            v_list.append( b1[i])

        for j in range(len(b2)):
            v_i = max(-A2[j] @ x , b2[j])
            v_list.append(v_i)
        return np.array(v_list)
    ###TODO###
    # Iterative reweighting algorithm

    A = np.vstack((A1, A2))
    b = np.hstack((b1, b2))
    x = np.zeros(A.shape[1])
    epsi = np.ones(len(b1) + len(b2)) * 2000
    epsi_new = epsi
    epsi_hat = epsi
    epsi_hat_new = epsi_hat
    M = 1e3
    gama = 0.15
    eta = 0.7
    sigma = 1e-3
    sigma_dot = 1e-4
    w = 1
    for _ in range(10000):
        W_arr = W(x, epsi, w)
        
        W_tmp = np.diag(W_arr)
        v = V(x)

        H_ATWA = A.T @ W_tmp @ A + H
        g_ATWv = A.T @ W_tmp @ v + g
        
        x_new = np.linalg.solve(H_ATWA, -g_ATWv)

        q_tmp = A @ (x_new - x)
        r_tmp = (1 - v) *  (A @ x + b)

        # print("W_arr: ", W_arr)
        # print("epsi: ", epsi)
        print("value", 0.5 * x_new.T @ H @ x_new + g.T @ x_new)
        update = True
        for __ in range(len(b)):
            if np.linalg.norm(q_tmp[__]) > M * np.power(np.linalg.norm(r_tmp[__]) ** 2 + epsi[__] ** 2,(0.5 + gama)):
                update = False
                break
        

        if update:
            # epsi_hat_new = [np.random.uniform(1e-9, eta * epsi_hat[__] + 1e-9) for __ in range(len(b))]
            epsi_hat_new = eta * epsi_hat
            # print("epsi_hat_new: ", epsi_hat_new)
            epsi_new = epsi_hat_new.copy()
            for __ in range(len(b2)):
                if min(A2[__] @ x + b2[__], 0) <= - epsi_hat[len(b1) + __]:
                    epsi_new[len(b1) + __] = epsi[len(b1) + __]
        else:
            epsi_hat_new = epsi_hat
            epsi_new = epsi
        # print("epsi_hat: ", epsi_hat)
        if (np.linalg.norm(x_new - x) < sigma) and (np.linalg.norm(epsi_hat) < sigma_dot):
            x = x_new
            break

        
        x = x_new

        epsi = epsi_new
        epsi_hat = epsi_hat_new
        # w += 5

    return x

if __name__ == '__main__':
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-12, -13.2])
    A1 = np.array([[1, 1]])
    b1 = np.array([-1])
    A2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([-2, -2])

    x = QP_solver(A1,A2,b1,b2,g,H)
    print(x)