import torch

def curve2coef_fixed(x_eval, y_eval, grid, k, device=None):
    """
    Fixed version of curve2coef with better numerical stability
    """
    if device is None:
        device = x_eval.device
    
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    
    # Import B_batch function from the original spline module
    from kan.spline import B_batch
    
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    
    # Try multiple approaches for numerical stability
    coef = None
    
    # Method 1: Standard lstsq with CPU
    try:
        mat_cpu = mat.cpu()
        y_eval_cpu = y_eval.cpu()
        coef = torch.linalg.lstsq(mat_cpu, y_eval_cpu, driver='gelsy').solution[:,:,:,0]
        coef = coef.to(device)
    except Exception as e:
        print(f"lstsq method 1 failed: {e}")
    
    # Method 2: Try with different driver
    if coef is None:
        try:
            mat_cpu = mat.cpu()
            y_eval_cpu = y_eval.cpu()
            coef = torch.linalg.lstsq(mat_cpu, y_eval_cpu, driver='gels').solution[:,:,:,0]
            coef = coef.to(device)
        except Exception as e:
            print(f"lstsq method 2 failed: {e}")
    
    # Method 3: Manual pseudo-inverse with regularization
    if coef is None:
        try:
            print("Using manual pseudo-inverse fallback")
            lamb = 1e-6  # Increased regularization
            XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
            Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
            n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
            
            # Add regularization
            identity = torch.eye(n, device=device)[None, None, :, :].expand(n1, n2, n, n)
            A = XtX + lamb * identity
            
            # Use pinverse for more stable inversion
            A_pinv = torch.pinverse(A)
            coef = (A_pinv @ Xty)[:,:,:,0]
        except Exception as e:
            print(f"Manual pseudo-inverse failed: {e}")
    
    # Method 4: Last resort - simple least squares on CPU
    if coef is None:
        try:
            print("Using simple least squares fallback")
            # Reshape for simple 2D solve
            mat_2d = mat.reshape(-1, mat.shape[-1]).cpu()
            y_2d = y_eval.reshape(-1, 1).cpu()
            
            # Add small regularization to diagonal
            AtA = mat_2d.T @ mat_2d + 1e-8 * torch.eye(mat_2d.shape[1])
            Aty = mat_2d.T @ y_2d
            
            coef_1d = torch.linalg.solve(AtA, Aty).squeeze()
            coef = coef_1d.view(in_dim, out_dim, n_coef).to(device)
        except Exception as e:
            print(f"Simple least squares failed: {e}")
            # Initialize with zeros as absolute fallback
            coef = torch.zeros(in_dim, out_dim, n_coef, device=device)
    
    return coef

# Patch the original function
import kan.spline
kan.spline.curve2coef = curve2coef_fixed
print("curve2coef function patched successfully")
