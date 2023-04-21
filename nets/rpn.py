import  numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from torchvision.ops import nms 
from utils.anchors import _enumerate_shifted_anchor , generate_anchor_base
from utils.utils_bbox import loc2bbox

class ProposalCreator():
    def __init__(self,
                 mode,
                 nms_iou = 0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms = 600,
                 n_test_pre_nms = 3000,
                 n_test_post_nms = 300,
                 min_size = 16
                ):
        self.mode = mode
        self.nms_iou = nms_iou
        self.n_train_post_nms = n_train_post_nms
        self.n_train_pre_nms = n_train_pre_nms

        self.n_test_post_nms= n_test_post_nms
        self.n_test_pre_nms  = n_test_pre_nms
        self.min_size = min_size

    def __call__(self, loc, score,anchor, img_size, scale =1.): 
        if self.mode == "training":
            n_pre_nms  = self.n_train_pre_nms
            n_post_nms = self.n_test_post_nms
        else :
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将anchor锚点（也称为先验框） 转换为tensor, numpy==> tensor
        anchor = torch.from_numpy(anchor).type_as(loc)

        # 将每个anchor转换为初步建议的box （建议框）， 具体可以查看box的格式（分别是左上角与右上角的横纵坐标  （X1,Y1, X2, Y2） 的格式）
        roi = loc2bbox(anchor,loc)

        # 防止建议框超出图像的边缘,img_size[1]是图像的宽

        roi[:,[0,2]] = torch.clamp(roi[:,[0,2]],min =0 , max = img_size[1])
        roi[:,[1,3]] = torch.clamp(roi[:,[1,3]],min =0 , max = img_size[0])

        # 建议框的宽高最小值不可以小于16 
        min_size = self.min_size * scale
        keep = torch.where((roi[:,2] - roi[:,0]) >=min_size & ((roi[:,3] - roi[;,1]) >= min_size))[0]
         #[0]是因为输出为([x,x,x],)的tuple格式

        # 将keep后的roi筛选出来
        roi = roi[keep,:]
        score = score[keep]

        # 下面根据得分进行排序，取出符合条件的box
        order = torch.argsort(score,descending=True)
        #如果在非极大抑制前，规定了建议框的数量，则进一步提取相应数量的建议框
        if n_pre_nms >0 : 
            order == order[:n_pre_nms]
        roi = roi[order,:]
        score = score[order]

        # 对建议框进行非极大抑制
        keep = nms(roi,score,self.nms_iou)
        # 如果符合条件的建议框数量少于post_nms 的要求，则随机重复提取建议框满足要求
        if len(keep) < n_post_nms:
            index_entra = np.random.choice(range(len(keep)),size = (n_post_nms-len(keep)),replace=True)
            keep = torch.cat([keep,keep[index_entra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi
    
class RegionProposalNetwork(nn.Module):
    # 512在这里只是默认值，在具体的调用中可以自己修改
    def __init__(self,
                 in_channel = 512,
                 mid_channel = 512,
                 ratios = [ 0.5,1,2],
                 anchor_scales = [8,16,32],
                 feat_stride = 16,
                 mode = "training"
                 ) -> None:
        super().__init__()
        # 生成基础的 anchor_base 一般为九个，shape为[9,4]
        self.anchor_base = generate_anchor_base(anchor_scale = anchor_scales)
        n_anchor = self.anchor_base.shape[0]

        # 根据原文中，我们先进行3*3的特征卷积
        self.conv1 = nn.Conv2d(in_channel,mid_channel,3,1,1)

        # 进行分类预测 
        self.classify =  nn.Conv2d(mid_channel,n_anchor*2, 1,1,0)

        #进行回归预测
        self.loc = nn.Conv2d(mid_channel,n_anchor*4,1,1,0)

        #特征点的间距，即原图到共享特征层 这个过程的缩放的大小
        self.feat_stride = feat_stride

        #实例化建议框生成器
        self.proposal_layer = ProposalCreator(mode)

        # 对卷积网络进行权值初始化
        normal_init(self.conv1,0,0.01)
        normal_init(self.score,0,0.01)
        normal_init(self.loc,0,0.01)

    def forward(self,x,img_size,scale=1.):
        # 我的理解c为channel数， 此时应该是共享特征层的通道数 默认vgg 512 channel
        n,c,h,w = x.shape

        # 连接上述定义的卷积层
        x = F.relu(self.conv1(x))
        # 回归
        rpn_locs  = self.loc(x)
        rpn_locs =  rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        # 判断，分类 positive or negative
        rpn_scores = self.classify(x)
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous().view(n,-1,2)

        # 对分类的结果进行softmax回归
        rpn_softmax_score = F.softmax(rpn_scores,dim = -1)
        rpn_fg_scores = rpn_softmax_score[:,:,1].contiguous()
        # 这一步不理解，如果做出了这样的变换，那score将与其他的元素混在一起
        rpn_fg_scores = rpn_scores.view[n,-1]

        # 根据anchor_base生成全局的anchors 也叫先验框 （38*38*9，4）
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride,h,w)
        rois = []
        roi_indices = []

        # 对batch进行逐一操作
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i],rpn_fg_scores[i],anchor,img_size,scale=scale)
            batch_index = i *torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))
        
        rois = torch.cat(rois,dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices,dim =0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    

def normal_init(m,mean,stddev,truncated = False):
        # 在使用m.weight.data.normal_()函数随机初始化权重时，如果没有指定std（即标准差），
        # 则会使用默认值为1的标准正态分布（均值为0，标准差为1）来初始化权重，也就是说权重的原始stddev是1。
        # 因此，在执行.mul_(stddev)操作时，实际上是将权重缩放为标准差为stddev的正态分布。
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


        
        
        
        



