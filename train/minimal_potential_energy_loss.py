import torch
from torch_scatter import scatter_add





class MPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,U,K,F):

        with torch.no_grad():
            ctx.save_for_backward(U)
            energy, grad= MPE_loss_func(K,F,U)
            ctx.gradient = grad
            output = energy
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.gradient
        return grad, None, None

@torch.no_grad()
def MPE_loss_func(K,F,u):
    res = K.shape[0]
    device = K.device
    K = K.reshape(res,res,res,27,3,3)
    xs = torch.arange(res,device=device)
    ys = torch.arange(res,device=device)
    zs = torch.arange(res,device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
    coo = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # 形状: (n^3, 3)
    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    adj_coo = (coo.unsqueeze(0) + offsets.view(27,1,3)+ res) % res
    adj_coo = adj_coo.reshape(-1,3)

    adj_u = u[adj_coo[:,0],adj_coo[:,1],adj_coo[:,2],:].view(27,-1,3).transpose(0,1).reshape(res,res,res,27,3)
    Ku=torch.einsum('xyznij,xyznj->xyzi',K,adj_u)

    grad = Ku - F
    
    loss = torch.einsum('xyzj,xyzj->xyz',u,0.5*Ku - F) 

    return torch.mean(loss), grad


def displacement_regularization(U,coo,batch_size):
    loss_U=torch.zeros(3,requires_grad=True,device=U.device)
    for i in range (batch_size):
        mask=coo[:,0]==i
        u_=U[mask].reshape(-1,3)
        loss_U=loss_U+torch.mean(u_,dim=0)**2
    return torch.mean(loss_U)



def get_node_type(voxel):
    """
    According to the 8 ocuraries around the node, the node type is judged
        Args:
            voxel shape(n , n , n)
            0,1 indicates whether it is a solid voxel
            
        Returns:
            voxel (long)
            shape(n , n , n)
                [0,255]indicates node type
    """
    node_type=torch.zeros(voxel.shape,dtype=torch.int,device=voxel.device)
    r=voxel.shape[0]
    soile_voxel_coo=torch.nonzero(voxel,as_tuple=False).to(voxel.device)
    hex=torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]],dtype=torch.int,device=voxel.device)
    tips=1
    for i in range (0,8):
        coo=((soile_voxel_coo+hex[i]+r)%r).to(voxel.device)
        node_type[coo[:,0],coo[:,1],coo[:,2]]+=tips
        tips=tips*2
    return node_type.long()


def loss_res(K,F,u,coo_vail):
    res = K.shape[0]
    device = K.device
    K = K.reshape(res,res,res,27,3,3)
    xs = torch.arange(res,device=device)
    ys = torch.arange(res,device=device)
    zs = torch.arange(res,device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
    coo = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # 形状: (n^3, 3)
    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    adj_coo = (coo.unsqueeze(0) + offsets.view(27,1,3)+ res) % res
    adj_coo = adj_coo.reshape(-1,3)

    adj_u = u[adj_coo[:,0],adj_coo[:,1],adj_coo[:,2],:].view(27,-1,3).transpose(0,1).reshape(res,res,res,27,3)
    Ku=torch.einsum('xyznij,xyznj->xyzi',K,adj_u)

    grad = Ku - F
    grad = grad[coo_vail[:,1],coo_vail[:,2],coo_vail[:,3],:]
    # print(coo_vail.shape)
    # print(torch.linalg.norm(grad).item())
    return torch.linalg.norm(grad)
    return torch.sum(grad*grad)

def MPE_loss(optimizer,batch,u_list,K_list,F_list,weight):
    
    mpe_loss = MPE.apply
    # batch["coord"].int()
    loss_dict = {}
    loss = 0
    i = len(u_list) - 1 
    # for i in range(len(u_list)):
    loss_level = loss_res(K_list[i],F_list[i],u_list[i],batch["coord"].int())
    loss_dict[f"level-{i}"] = loss_level
    loss = loss + loss_level
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_dict