import torch
import matplotlib.cm as cm
import numpy as np


def vis_node(coo,fea):
    fea=fea.detach().cpu().numpy()
    coo=coo.detach().cpu().numpy()
    with open('paddingnode.obj', 'w') as f:
        for j in range (0,coo.shape[0]):
            print('v ',coo[j][0],coo[j][1],coo[j][2],
                fea[j][0],fea[j][1],fea[j][2],file=f)
    


def loss_voxel(voxel,path):
    coo=torch.nonzero(voxel[0], as_tuple=False)
    coo=coo.detach().cpu().numpy()
    with open(path+'/'+'voxel.obj', 'w') as f:
        for j in range (0,coo.shape[0]):
            print('v ',coo[j][0],coo[j][1],coo[j][2],
                1.,0.,0.,file=f)
    exit()
def loss_vis(U_p,U,voxel,path): 
    base_coo=torch.nonzero(voxel, as_tuple=False)
    base_coo=base_coo[base_coo[:, 0] == 0]
    hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
    for i in hex:
        coo=(base_coo[:,1:]+i)%64
        print(coo.shape,base_coo.shape)
        voxel[:,coo[:, 0], coo[:, 1], coo [:, 2]] = 1
    
    
    base_coo=torch.nonzero(voxel, as_tuple=False)
    base_coo=base_coo[base_coo[:, 0] == 0].float()
    coo=base_coo.clone()
    feature = torch.sum(torch.abs(U_p.features_at_coordinates(coo)-U.features_at_coordinates(base_coo)),dim=1)
    feature=(feature - torch.min(feature)) * 1.0 / (torch.max(feature) - torch.min(feature))
    coo=base_coo
    feature=feature.detach().cpu().numpy()
    coo=coo.detach().cpu().numpy()
    cmap = cm.get_cmap('jet',coo.shape[0])
    color= cmap(feature)
    with open(path+'/'+'solidnode.obj', 'w') as f:
        for j in range (0,coo.shape[0]):
            print('v ',coo[j][1],coo[j][2],coo[j][3],
                color[j][0],color[j][1],color[j][2],file=f)
            
    
def loss_res(residual,voxel,path): 
    
    base_coo=torch.nonzero(voxel, as_tuple=False)
    base_coo=base_coo[base_coo[:, 0] == 0]
    hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
    for i in hex:
        coo=(base_coo[:,1:]+i)%64
        voxel[:,coo[:, 0], coo[:, 1], coo [:, 2]] = 1
    coo=torch.nonzero(voxel, as_tuple=False)
    
    res_fea=residual.features_at_coordinates(coo.contiguous().float())
    feature = torch.sum(torch.abs(res_fea),dim=1)
    res_max= torch.max(feature)
    res_min= torch.min(feature)
    feature=(feature - torch.min(feature)) * 1.0 / (torch.max(feature) - torch.min(feature))
    feature=feature.detach().cpu().numpy()
    coo=coo.detach().cpu().numpy()
    cmap = cm.get_cmap('jet',coo.shape[0])
    color= cmap(feature)
    with open(path+'/'+'residual.obj', 'w') as f:
        print('# max residual:\t',res_max.detach().cpu().numpy(),'\t min residual:\t',res_min.detach().cpu().numpy())
        for j in range (0,coo.shape[0]):
            print('v ',coo[j][1],coo[j][2],coo[j][3],color[j][0],color[j][1],color[j][2],file=f)
    
def dis_vis(model): 
    for i in range(0,6):
        U=model.F.reshape(-1,6,3)[:,i,:]
        U=torch.linalg.norm(U,dim=1)
        U=(U - torch.min(U)) * 1.0 / (torch.max(U) - torch.min(U)).detach()
        U=U.cpu().numpy()
        coo=model.C[:,1:].detach().cpu().numpy()
        cmap = cm.get_cmap('jet',coo.shape[0])
        color= cmap(U)
        with open(str(i)+'_down_dis_GT_node.obj', 'w') as f:
            for j in range (0,coo.shape[0]):
                print('v ',coo[j][0],coo[j][1],coo[j][2],
                    color[j][0],color[j][1],color[j][2],file=f)
        
