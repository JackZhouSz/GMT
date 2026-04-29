import torch
import time
import spconv.pytorch as spconv
from train.minimal_potential_energy_loss import MPE_loss
from train._utils import calculate_residual_batch
import matplotlib.cm as cm
@torch.no_grad()   


def validate(net,batch,Ke,Fe,F,smooth,batch_size):
    """computer batch error between predicated homogenized tensor and the groundtruth

    Args:
        output (torch.cuda.FloatTensor): predicated displacement,it should be batch_size*18*N*N*N
        
        Ke (torch.cuda.FloatTensor): element stiffness matrics, it should be batch_size*24*24
        Fe (torch.cuda.FloatTensor): element force matrics, it should be batch_size*24*6
        X0 (torch.cuda.FloatTensor): element force matrics, it should be batch_size*24*6

    Returns:
        batch_error_mean,
        batch_error,
        batch_loss_actual,
        batch_predict
    """
    device=Ke.device

    # net.train()
    # F=torch.cat(F_list,dim=0)
    U=net(batch,Ke,F,smooth,batch_size)
    # Ugt=net(batch,Ke,F,500,batch_size)
    # feature=Ugt.features-U.features
    # vis_res(Ugt.indices,feature)
    # # print(feature.dtype)
    # exit()
    # print(torch.max(feature,dim=0),torch.min(feature,dim=0))

    # coo,feature=batch["coord"].int(),batch["feature"].float()

    #     feature.requires_grad_(True)

    #     sparse_tensor = spconv.SparseConvTensor(
    #         features=feature,
    #         indices=coo,
    #         spatial_shape=[self.r,self.r,self.r],
    #         batch_size=batch_size
    #     )

    # print("sdas",torch.linalg.norm(calculate_residual_(U,K_list,F,batch_size)))
    # exit()
    # output=U.dense()
    residual=torch.linalg.norm(calculate_residual_batch(U,batch,Ke,F,batch_size))

    validate_dict={}
    validate_dict["residual"]=residual


    return validate_dict


def vis_res(coo,fea):
    fea2 = (fea - torch.min(fea))/(torch.max(fea)-torch.min(fea))
    coo = coo.cpu().numpy()
    cmap = cm.get_cmap("jet")   
    for i in range (0,6):
        for j in range (0,3):

            # abs
            fea_ij=fea[:,i*3+j]
            fea_ij = torch.abs(fea_ij).cpu().numpy() * 10
            color = cmap(fea_ij)[..., :3] 
            with open(f'./res_vis/abs-condition-{i}-direction-{j}.obj', 'w') as f:
                for k in range (0,coo.shape[0]):
                    print('v ',coo[k][1],coo[k][2],coo[k][3],color[k][0],color[k][1],color[k][2],file=f)

            # related
            fea_ij = fea2[:,i*3+j].cpu().numpy()
            # fea_ij = ((fea_ij + 1)/2).cpu().numpy() * 10
            color = cmap(fea_ij)[..., :3] 
            with open(f'./res_vis/related-condition-{i}-direction-{j}.obj', 'w') as f:
                for k in range (0,coo.shape[0]):
                    print('v ',coo[k][1],coo[k][2],coo[k][3],color[k][0],color[k][1],color[k][2],file=f)
    print(torch.max(fea,dim=0))
    print(torch.min(fea,dim=0))
    # return rgb.to(x.device, x.dtype)