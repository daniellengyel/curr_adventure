# Finite Difference 
def FD_2E0(F, X, h, dim):
    hess = []
    all_out_central = F.f(X)
    x = X[0]
    i = 0

    x_back = x - jnp.eye(x.shape[0]) * h
    x_forward = x + jnp.eye(x.shape[0]) * h
    
    out_back = F.f(x_back)
    out_forward = F.f(x_forward)
    
    curr_hess = jnp.diag((out_back + out_forward - 2 * all_out_central[i]) / h**2) 
    out_back = F.f(x_back)
    out_forward = F.f(x_forward)
    
    for d in range(dim):
    
        x_forward_forward = x + jnp.eye(x.shape[0]) * h + jnp.eye(x.shape[0], k=d) * h
        x_forward_backward = x + jnp.eye(x.shape[0]) * h - jnp.eye(x.shape[0], k=d) * h
        x_backward_forward = x - jnp.eye(x.shape[0]) * h + jnp.eye(x.shape[0], k=d) * h
        x_backward_backward = x - jnp.eye(x.shape[0]) * h - jnp.eye(x.shape[0], k=d) * h

        out_forward_forward = F.f(x_forward_forward)
        out_forward_backward = F.f(x_forward_backward)
        out_backward_forward = F.f(x_backward_forward)
        out_backward_backward = F.f(x_backward_backward)
        
        curr_off_diag = jnp.eye(x.shape[0], k=d) * (out_forward_forward - out_forward_backward - out_backward_forward + out_backward_backward).reshape(-1, 1)/(4.*h**2)
        curr_hess += curr_off_diag + curr_off_diag.T
        
    hess.append(curr_hess)
        
    return jnp.array(hess)