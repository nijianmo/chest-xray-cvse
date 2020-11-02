import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from torch.autograd import Variable
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False, pretrained_path=''):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.linear_frontal, self.linear_lateral = self.get_model() # define model
        self.activation = nn.ReLU()

    def get_model(self):
        model = None
        out_features = None
        avg_func = None
        
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2] # don't consider last two layers
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        
        elif self.model_name == 'densenet121':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
            
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        
        elif self.model_name == 'vgg19':
            vgg19 = models.vgg19(pretrained=self.pretrained)
            modules = list(vgg19.features.children())[:-2]
            model = nn.Sequential(*modules)
            # avg_func = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0) # B,512
            avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0) # B,512,8,8
            out_features = 512
        
        # these two layers are actually not used?
        linear_frontal = nn.Linear(in_features=out_features, out_features=out_features)
        linear_lateral = nn.Linear(in_features=out_features, out_features=out_features)
        
        return model, out_features, avg_func, linear_frontal, linear_lateral

    def forward(self, frontal_images, lateral_images):
        """
        :param images:
        :return:
        """
        
        # do not train backbone network
        visual_features_frontal = self.model(frontal_images)
        visual_features_lateral = self.model(lateral_images)
        
        # take average
        # avg_features = self.avg_func(visual_features).squeeze()
        avg_features_frontal = self.avg_func(visual_features_frontal)
        avg_features_lateral = self.avg_func(visual_features_lateral)
        
        batch_size, hidden_size = avg_features_frontal.shape[0], avg_features_frontal.shape[1]

        avg_features_frontal = avg_features_frontal.view(batch_size, hidden_size, -1)
        avg_features_lateral = avg_features_lateral.view(batch_size, hidden_size, -1)
        
        avg_features_frontal = self.linear_frontal(avg_features_frontal.transpose(1, 2))
        avg_features_lateral = self.linear_lateral(avg_features_lateral.transpose(1, 2))
        
        return visual_features_frontal, visual_features_lateral, avg_features_frontal, avg_features_lateral


class MLC(nn.Module):
    def __init__(self,
                 classes=1,
                 fc_in_features=2048):
        super(MLC, self).__init__()
        print('mlc fc_in_features={}'.format(fc_in_features))
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        preds = self.classifier(avg_features)
        # preds = F.sigmoid(preds)
        return preds
    
    
class Attn(nn.Module):
    def __init__(self, hidden_size=256):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, u, v):
        # u: B x T x H
        # v: B x H
        batch_size = u.size(0)
        seq_len = u.size(1)
     
        H = v.unsqueeze(1).repeat(1,seq_len,1) # [B,1,H] -> [B,T,H]
        
        attn_energies = F.softmax(self.score(u, H), -1) # B,T and normalize on T
       
        # l2 distance
        d = torch.norm(u - H, p=2, dim=-1) # B,T,H -> B,T
        # inner product
        #d = torch.mul(u, H).sum(-1)

        w_d = d * attn_energies
        w_d = w_d.sum(-1) # B,1
    
        return w_d, attn_energies

    def score(self, u, H):
       
        energy = self.attn(torch.cat([u, H], -1)) # [B,T,2*H] -> [B,T,H]
        energy = energy.view(-1, self.hidden_size) # [B*T,H]
        
        v = self.v.unsqueeze(1) # [H,1]
        energy = energy.mm(v) # [B*T,H] x [H,1] -> [B*T,1]
        
        att_energies = energy.view(u.size(0), -1) # [B,T] 
        return att_energies    
       
class FeatureTransform(nn.Module):
    def __init__(self, transform_dim, dropout=0.2, use_batchnorm=False, use_relu=False, use_linear=True, momentum=0.1):
        super(FeatureTransform, self).__init__()
        
        self.transform_dim = transform_dim
        self.use_batchnorm = use_batchnorm
        self.use_relu = use_relu
        self.use_linear = use_linear
        print('bn transform_dim = ', transform_dim) 
        
        # self.linear1 = nn.Linear(in_features=input_dim, out_features=transform_dim)
        self.bn = nn.BatchNorm1d(num_features=transform_dim, momentum=momentum)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=transform_dim, out_features=transform_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, feat):
        # feat = self.linear1(feat)
        # visual: B,WH,C
        # semantic: B,C
        if self.use_batchnorm:
            if len(feat.shape) == 3: # visual emb
                feat = feat.transpose(1, 2).contiguous()
                feat = self.bn(feat)
                feat = feat.transpose(1, 2).contiguous()
            else:
                feat = self.bn(feat)
        if self.use_relu:
            feat = self.relu(feat)
        feat = self.dropout(feat)
        if self.use_linear:
            feat = self.linear2(feat)
         
        feat = F.normalize(feat, p=2, dim=-1) # l2 normalize on the last dimention
        return feat
    
class RankModel(nn.Module):  
    def __init__(self, args=None, image_hidden_size=1024, sentence_hidden_size=700):
        super(RankModel, self).__init__()
        
        self.args = args
        # define modules
        self.transform_dim = self.args.transform_dim
        self.image_hidden_size = image_hidden_size
        self.sentence_hidden_size = sentence_hidden_size
        
        self.frontal_transform = nn.Linear(in_features=image_hidden_size, out_features=self.transform_dim)
        self.lateral_transform = nn.Linear(in_features=image_hidden_size, out_features=self.transform_dim)
        self.sentence_transform = nn.Linear(in_features=sentence_hidden_size, out_features=self.transform_dim)
        
        self.feat_transform = FeatureTransform(self.transform_dim, dropout=self.args.dropout, use_batchnorm=self.args.use_batchnorm, use_relu=self.args.use_relu, use_linear=self.args.use_linear, momentum=self.args.momentum)

        self.attn = Attn(hidden_size=self.transform_dim)
    
    def forward(self, u_frontal, u_lateral, v_sent, u_neg_frontal=None, u_neg_lateral=None, v_neg_sent=None, training=True):
        
        # images_frontal, images_lateral: B x C x 16 x 16 
        # sentence: B x C
        batch_size = u_frontal.shape[0]
        
        # B x C x HW -> B x HW x C
        u_frontal = u_frontal.view(batch_size, self.image_hidden_size, -1).transpose(1, 2).contiguous()
        u_lateral = u_lateral.view(batch_size, self.image_hidden_size, -1).transpose(1, 2).contiguous()
        
        # transform
        u_frontal = self.frontal_transform(u_frontal)
        u_lateral = self.lateral_transform(u_lateral)
        v_sent = self.sentence_transform(v_sent)
        
        u_frontal = self.feat_transform(u_frontal)
        u_lateral = self.feat_transform(u_lateral)
        v_sent = self.feat_transform(v_sent)
            
        # attention
        # (B x HW x C, B x C)
        u_frontal_v_d, u_frontal_v_attn = self.attn(u_frontal, v_sent)
        u_lateral_v_d, u_lateral_v_attn = self.attn(u_lateral, v_sent)
        
         # final distance metric
        u_v_d = 0.5 * (u_frontal_v_d + u_lateral_v_d)
      
        # if self.training:
        if training:
            # do the same for negatives
            # B x N x C x HW -> B*N x HW x C
            # B x N x C -> B*N x C
            
            u_neg_frontal = u_neg_frontal.view(batch_size * self.args.num_negatives, self.image_hidden_size, -1).transpose(1, 2).contiguous()
            u_neg_lateral = u_neg_lateral.view(batch_size * self.args.num_negatives, self.image_hidden_size, -1).transpose(1, 2).contiguous()
            v_neg_sent = v_neg_sent.view(batch_size * self.args.num_negatives, -1)

            u_neg_frontal = self.frontal_transform(u_neg_frontal)
            u_neg_lateral = self.lateral_transform(u_neg_lateral)
            v_neg_sent = self.sentence_transform(v_neg_sent)
            
            u_neg_frontal = self.feat_transform(u_neg_frontal)
            u_neg_lateral = self.feat_transform(u_neg_lateral)
            v_neg_sent = self.feat_transform(v_neg_sent)
            
            # replicate positive input
            u_frontal = u_frontal.unsqueeze(1).repeat(1, self.args.num_negatives, 1, 1)
            u_lateral = u_lateral.unsqueeze(1).repeat(1, self.args.num_negatives, 1, 1)
            v_sent = v_sent.unsqueeze(1).repeat(1, self.args.num_negatives, 1)
               
            u_frontal = u_frontal.view(batch_size * self.args.num_negatives, -1, self.transform_dim)
            u_lateral = u_lateral.view(batch_size * self.args.num_negatives, -1, self.transform_dim)
            v_sent = v_sent.view(batch_size * self.args.num_negatives, -1) 
            
            u_frontal_v_neg_d, u_frontal_v_neg_attn = self.attn(u_frontal, v_neg_sent)
            u_lateral_v_neg_d, u_lateral_v_neg_attn = self.attn(u_lateral, v_neg_sent)

            u_neg_frontal_v_d, u_neg_frontal_v_attn = self.attn(u_neg_frontal, v_sent)
            u_neg_lateral_v_d, u_neg_lateral_v_attn = self.attn(u_neg_lateral, v_sent)

            u_v_neg_d = 0.5 * (u_frontal_v_neg_d + u_lateral_v_neg_d)
            u_neg_v_d = 0.5 * (u_neg_frontal_v_d + u_neg_lateral_v_d)

            u_v_d = u_v_d.unsqueeze(1).repeat(1, self.args.num_negatives).view(-1)
            
            return u_v_d, u_v_neg_d, u_neg_v_d, u_frontal_v_attn, u_lateral_v_attn, u_frontal_v_neg_attn, u_lateral_v_neg_attn, u_neg_frontal_v_attn, u_neg_lateral_v_attn
    
        else:
            return u_v_d, u_frontal_v_attn, u_lateral_v_attn
    
    
if __name__ == '__main__':
    """python -m utils.rank_pair_models"""
    import torchvision.transforms as transforms

    import warnings
    warnings.filterwarnings("ignore")
    import argparse 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--transform_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_batchnorm', action='store_true', default=True)
    parser.add_argument('--use_relu', action='store_true', default=True)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--num_negatives', type=int, default=4)
    
    args = parser.parse_args()
    
    model = RankModel(args)
    
    batch_size = 2
    u_frontal = torch.randn((batch_size, 1024, 16, 16))
    u_lateral = torch.randn((batch_size, 1024, 16, 16))
    v_sent = torch.rand((batch_size, 700))
    u_neg_frontal = torch.randn((batch_size, 4, 1024, 16, 16))
    u_neg_lateral = torch.randn((batch_size, 4, 1024, 16, 16))
    v_neg_sent = torch.rand((batch_size, 4, 700))

    u_v_d, u_v_neg_d, u_neg_v_d, u_frontal_v_attn, u_lateral_v_attn, u_frontal_v_neg_attn, u_lateral_v_neg_attn, u_neg_frontal_v_attn, u_neg_lateral_v_attn = model(u_frontal, u_lateral, v_sent, u_neg_frontal, u_neg_lateral, v_neg_sent)
    
    print('u_v_d shape: ', u_v_d.shape)
    print('u_frontal_v_attn shape: ', u_frontal_v_attn.shape)
    print('u_v_neg_d shape: ', u_v_neg_d.shape)
    print('u_neg_v_d shape: ', u_neg_v_d.shape)

    
#     cam = torch.mul(visual_features, alpht_v.view(alpht_v.shape[0], alpht_v.shape[1], 1, 1)).sum(1)
#     cam.squeeze_()
#     cam = cam.cpu().data.numpy()
#     for i in range(cam.shape[0]):
#         heatmap = cam[i]
#         heatmap = heatmap / np.max(heatmap)
#         print(heatmap.shape)
