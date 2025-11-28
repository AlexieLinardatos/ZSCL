import torch

def svd_basis(gradient_per_layer_list, energy=0.97):

    basis_per_layer = {}
    
    for name, gradient_list in gradient_per_layer_list.items():
        
        if len(gradient_list) < 2:
            continue
        q = min(len(gradient_list), 20) 
        G = torch.stack(gradient_list, dim=0)
        G = G.cpu()
        # print(f"{name}: len = {len(gradient_list)}, G.shape = {G.shape}")
        if hasattr(torch.linalg, "svd_lowrank"):
            U, S, V_transpose = torch.linalg.svd_lowrank(G, q=q, niter=4)
        elif hasattr(torch, "svd_lowrank"):
            U, S, V_transpose = torch.svd_lowrank(G, q=q, niter=4)
        else:
            U, S, V_transpose = torch.linalg.svd(G, full_matrices=False)
        total_energy = S.pow(2).sum()
        running = 0.0
        k = 0
        for i in range(len(S)):
            running += S[i].pow(2)
            if running/total_energy >= energy:
                k = i+1
                break

        V_transpose = V_transpose.T
        # print(f"{name}: V.shape = {V_transpose.shape}, k={k}")
        basis_per_layer[name] = V_transpose[:k, :].contiguous().clone().cpu()
    return basis_per_layer



gradients = torch.load("grad_DTD.pth", weights_only=False)

# basis_per_layer = svd_basis(gradients)
# torch.save(basis_per_layer, "grad_DTD.pth")

for key, grad in gradients.items():
    # print(f"{key}:{(grad)}")
    print(grad[-1])
# print((gradients[0]))
# print(gradients[-1].detach().cpu().flatten())