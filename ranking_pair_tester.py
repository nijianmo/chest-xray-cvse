import argparse
import pickle
import random
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.rank_pair_models import *
from utils.rank_pair_dataset import *
from utils.rank_image_dataset import *
from utils.loss import *
from utils.logger import Logger

from tqdm import tqdm
from apex import amp

def set_seed(seed, gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)
    print('Set seeds to {}'.format(seed))
    
    
class DebuggerBase:
    def __init__(self, args):
        self.args = args
        
        self._init_model_path()
        self.model_dir = self._init_model_dir()
        
        if not self.args.debug:
            self.writer = self._init_writer()
        
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()

        print('create train_data_loader')
        self.load_data()
        
        print('create test_image_dataloader')
        self.test_image_dataloader = self._init_image_data_loader(self.args.test_image_file_list)
        
        print('create model')
        self.model = RankModel(args)

        if self.args.cuda:
            self.model = self.model.cuda()
        
        print('create optimizer')
        # init optimizer        
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        if not self.args.debug:
            self.logger = self._init_logger()
        
        if not self.args.debug:
            self.writer.write("{}\n".format(self.args))
        
        # load model
        self.model_state_dict = self._load_mode_state_dict()
    
        # mix-precision
        if self.args.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # metrics
        self.min_MAP_3 = float('-inf')
        self.min_Recall_3 = float('-inf')
        self.min_train_loss = float('inf')
    
    def load_data(self):
        
        # load image embedding with size 1024 x 16 x16
        print('load image embeddings')
        image_embeddings_path = os.path.join(self.args.data_path, self.args.image_embeddings_path)
        self.image_embeddings = torch.load(image_embeddings_path)
        
        num_clusters = self.args.num_clusters
        print('load kmeans_{}'.format(num_clusters))
        with open('kmeans_{}.pkl'.format(num_clusters), 'rb') as f:
            _, self.centers = pickle.load(f)  
        self.centers = torch.tensor(self.centers)
               
        # pass to dataloader
        self.groups_path = 'report2group_{}.pkl'.format(num_clusters)
    
        # pass to image dataloader
        print('load report2group{}.pkl'.format(num_clusters))
        with open('report2group_{}.pkl'.format(num_clusters), 'rb') as f:
            _, _, _, self.train_report2group, _ = pickle.load(f)
        print('load report2group_dev_test_{}.pkl'.format(num_clusters))
        with open('report2group_dev_test_{}.pkl'.format(num_clusters), 'rb') as f:
            _, _, _, self.dev_test_report2group, _ = pickle.load(f)
            
    def train(self):
        # start epoch is used for continual training
        for epoch_id in range(self.start_epoch, self.args.epochs):
            train_loss = self._epoch_train()
           
            if self.args.mode == 'train':
                # ReduceLROnPlateau scheduler needs the loss to decide whether to decrease lr or not
                self.scheduler.step(train_loss)

            Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5 = self._epoch_val()
            # print(train_loss, Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5)
            print("Epoch {}: Train loss={}, Recall_3={:.3f}, Recall_5={:.3f}, Precision_3={:.3f}, Precision_5={:.3f}, MAP_3={:.3f}, MAP_5={:.3f}".format(epoch_id, train_loss, Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5))
            
            self._save_model(epoch_id,
                             Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5,
                             train_loss)
            self._log(Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5,
                      train_loss=train_loss,
                      lr=self.optimizer.param_groups[0]['lr'],
                      epoch=epoch_id)
            self.writer.write("Epoch {}: Train loss={}, Recall_3={:.3f}, Recall_5={:.3f}, Precision_3={:.3f}, Precision_5={:.3f}, MAP_3={:.3f}, MAP_5={:.3f}\n".format(epoch_id, train_loss, Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5))
            
    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        if self.args.model_dir != '':
            return self.args.model_dir
        
        model_dir = os.path.join(self.args.model_path)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not self.args.debug: # only make new dirs for training
            model_dir = os.path.join(model_dir, self._get_now())

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        return model_dir

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            # if load model path exists, then load the model
            # continue training
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            self.model.load_state_dict(model_state['model'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            
            if not self.args.debug:
                self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
                self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            else:
                print("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
                print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            if not self.args.debug:
                self.writer.write("[Load Model Failed] {}\n".format(err))
            else:
                print("[Load Model Failed] {}\n".format(err))
            return None

    def _init_data_loader(self):
        
        def get_loader(file_list,
                       image_embeddings,
                       groups_path,
                       sentence_embeddings_path,
                       transform=None,
                       dataset_name='openi',
                       batch_size=32,
                       shuffle=False,
                       n_gpus=1,
                       num_workers=1):
            dataset = ChestXrayDataSet(file_list,
                                       image_embeddings,
                                       groups_path,
                                       sentence_embeddings_path=sentence_embeddings_path,
                                       transforms=transform,
                                       dataset_name=dataset_name,
                                       use_center=self.args.use_center,
                                       centers=self.centers,
                                       num_negatives=self.args.num_negatives)

            # random sampler
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      sampler=sampler,
                                                      num_workers=num_workers,
                                                      batch_size=batch_size,
                                                      collate_fn=collate_fn,
                                                      pin_memory=True)

            return data_loader
   

        # get training pairs
        file_list = os.path.join(self.args.data_path, self.args.train_image_file_list)
        sentence_embeddings_path = os.path.join(self.args.data_path, self.args.sentence_embeddings_path)
        
        data_loader = get_loader(file_list,
                                 self.image_embeddings,
                                 self.groups_path,
                                 sentence_embeddings_path,
                                 transform=None,
                                 dataset_name='mimic',
                                 batch_size=self.args.batch_size,
                                 n_gpus=self.args.n_gpus,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        
        return data_loader

    def _init_image_data_loader(self, file_list):
        
        def get_loader(file_list,
                       image_embeddings,
                       report2group,
                       num_clusters,
                       transform=None,
                       dataset_name='openi',
                       batch_size=32,
                       shuffle=False,
                       n_gpus=1,
                       num_workers=1):
            dataset = ChestXrayImageDataSet(file_list,
                                            image_embeddings,
                                            report2group,
                                            num_clusters,
                                            transforms=transform,
                                            dataset_name=dataset_name)

            # random sampler
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      sampler=sampler,
                                                      num_workers=num_workers,
                                                      batch_size=batch_size,
                                                      collate_fn=collate_fn_image,
                                                      pin_memory=True)

            return data_loader
        
        file_list = os.path.join(self.args.data_path, file_list)
        
        if 'train' in file_list:
            report2group = self.train_report2group
        else:
            report2group = self.dev_test_report2group
        
        data_loader = get_loader(file_list,
                                 self.image_embeddings,
                                 report2group,
                                 num_clusters=self.args.num_clusters,
                                 transform=None,
                                 dataset_name='mimic',
                                 batch_size=self.args.eval_batch_size,
                                 n_gpus=self.args.n_gpus,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return data_loader

    def _init_optimizer(self):
        
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate) 

    def _log(self,
             Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5,
             train_loss,
             lr,
             epoch):
        info = {
            'Recall_3': Recall_3, 
            'Recall_5': Recall_5, 
            'Precision_3': Precision_3, 
            'Precision_5': Precision_5, 
            'MAP_3': MAP_3, 
            'MAP_5': MAP_5,
            'train loss': train_loss,
            'learning rate': lr
        }

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        if not requires_grad:
            x.requires_grad = False
        return x

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)       
            
    def _save_model(self,
                    epoch_id,
                    Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5,
                    train_loss):
        def save_whole_model(_filename):
            if not self.args.debug:
                self.writer.write("Saved Model in {}\n".format(_filename))
                
            torch.save({'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

#         if MAP_3 > self.min_MAP_3:
#             file_name = "val_best_loss.pth.tar"
#             save_whole_model(file_name)
#             self.min_MAP_3 = MAP_3
        if Recall_3 > self.min_Recall_3:
            file_name = "val_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_Recall_3 = Recall_3

        if train_loss < self.min_train_loss:
            file_name = "train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss

class RankDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

    def triple_loss(self, d_a_p, d_a_n, margin=0.2) : 
        distance = d_a_p - d_a_n + margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
        
    def _epoch_train(self):
        
        self.model.train()
        num_batch = len(self.train_data_loader)
        print('start training... total {} batches...'.format(num_batch))

        loss = 0   
        # for u, v, u_neg, v_neg, u_frontal, u_lateral, v_sent, u_neg_frontal, u_neg_lateral, v_neg_sent in tqdm(self.train_data_loader):
        
        with tqdm(enumerate(self.train_data_loader)) as pbar:
            for i, (u, v, u_neg, v_neg, u_frontal, u_lateral, v_sent, u_neg_frontal, u_neg_lateral, v_neg_sent) in pbar:
                pbar.set_description('Batch {}'.format(i))
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                
                u_frontal = self._to_var(u_frontal)
                u_lateral = self._to_var(u_lateral)
                v_sent = self._to_var(v_sent)
                u_neg_frontal = self._to_var(u_neg_frontal)
                u_neg_lateral = self._to_var(u_neg_lateral)
                v_neg_sent = self._to_var(v_neg_sent)

                # model forward
                u_v_d, u_v_neg_d, u_neg_v_d, u_frontal_v_attn, u_lateral_v_attn, u_frontal_v_neg_attn, u_lateral_v_neg_attn, u_neg_frontal_v_attn, u_neg_lateral_v_attn = self.model(u_frontal, u_lateral, v_sent, u_neg_frontal, u_neg_lateral, v_neg_sent)

                # margin ranking loss

                batch_loss = self.triple_loss(u_v_d, u_v_neg_d, self.args.margin) + self.triple_loss(u_v_d, u_neg_v_d, self.args.margin)

                self.optimizer.zero_grad()

                if self.args.fp16:
                    with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    batch_loss.backward()

                self.optimizer.step()
                cur_loss = batch_loss.cpu().item()
                loss += cur_loss
                
                pbar.set_postfix(loss=cur_loss)
                pbar.update(0.1)
                
                if self.args.debug_sample and i > num_batch * self.args.debug_sample:
                    break
          
        # print distances of last batch 
        print("u_v_d: ")
        print(u_v_d[:10])
        print("u_v_neg_d: ")
        print(u_v_neg_d[:10])
        print("u_neg_v_d: ")
        print(u_neg_v_d[:10])
        
        return loss / i
        # return loss / num_batch

    def _epoch_val(self):
        # evaluate on both train and val set
        # check the topk group for each report
        self.model.eval()
        num_batch = len(self.dev_image_dataloader)
        print('start eval... total {} batches...'.format(num_batch))
        
        hits_ground_3, hits_3, K_3 = 0, 0, 0
        hits_ground_5, hits_5, K_5 = 0, 0, 0
        AP_3, AP_5 = 0, 0
        n_samples = 0
        with torch.no_grad():
        
            for i, (u, u_frontal, u_lateral, label) in tqdm(enumerate(self.dev_image_dataloader)):
                batch_size = u_frontal.shape[0]
                u_frontal = self._to_var(u_frontal)
                u_lateral = self._to_var(u_lateral)
                label = self._to_var(label) # one hot label
                
                total_d = [] # num_clusters x B
                for v_sent in self.centers:
                    v_sent = v_sent.unsqueeze(0).repeat(batch_size, 1) # B x Hs 
                    v_sent = self._to_var(v_sent)
                    # set training to false
                    u_v_d, u_frontal_v_attn, u_lateral_v_attn = self.model(u_frontal, u_lateral, v_sent, training=False) 
                    total_d.append(-1 * u_v_d) # pick the shortest as topk, so multiply by -1
                # [B,B,B...,B] -> num_clusters x B -> B x num_clusters
                total_d = torch.stack(total_d, 0).transpose(0, 1) 
                total_d = self._to_var(total_d)
                
                hit_ground_3, hit_3, ap_3, k_3 = self.hr_k(label, total_d, k=3)
                hits_ground_3 += hit_ground_3
                hits_3 += hit_3
                K_3 += k_3
                AP_3 += ap_3
                
                hit_ground_5, hit_5, ap_5, k_5 = self.hr_k(label, total_d, k=5)
                hits_ground_5 += hit_ground_5
                hits_5 += hit_5
                K_5 += k_5
                AP_5 += ap_5
                
                n_samples += self.args.batch_size
                
                if self.args.debug_sample and i > num_batch * self.args.debug_sample:
                    break
            
            Recall_3 = hits_3 / hits_ground_3
            Recall_5 = hits_5 / hits_ground_5
            Precision_3 = hits_3 / K_3
            Precision_5 = hits_5 / K_5
            MAP_3 = AP_3 / n_samples
            MAP_5 = AP_5 / n_samples
            
            return Recall_3, Recall_5, Precision_3, Precision_5, MAP_3, MAP_5
            
            
    def hr_k(self, target_onehot, pred, k=10):
        
        batch_size = pred.size(0)
        top_k = torch.zeros(batch_size, self.args.num_clusters, dtype=torch.bool).cuda()

        target_onehot = target_onehot.bool()
        # obtain the topk indices (B * K)
        _, indices = pred.topk(k=k) 
        # (B * cluster_size)
        _ = top_k.scatter_(1, indices, True) 

        # print(f"indices dim: {indices.size()}, indices total elements number: {indices.numel()}")
        # print(f"top_k dim: {top_k.size()}, top_k # of positive items: {top_k.sum().item()}")

        intersection = top_k & target_onehot
        # calculate the hit and ground size
        hit_size = intersection.sum().item() # hit size is the # of intersection
        ground_size = target_onehot.sum().item() # ground size is the # of positives in target 
        # print(f"intersection dim: {intersection.size()}, intersection # of positive items (hits): {hit_size}")
        # print(f"batch hit ratio: {hit_size / ground_size}")
        # print(f"prediction size (k*batch): {k*batch_size}, batch precision_k: {hit_size/(k*batch_size)}")

        inter_sum1 = intersection.sum(1) # the number of intersection items in each sample
        inter_sum1[inter_sum1==0] = 1 # avoid zero division in samples without intersection items
        # calculate the mean average precision
        pred_acc = top_k.cumsum(1)
        pred_acc[~top_k] = 1 # to avoid zero division for those negative samples
        inter_acc = intersection.cumsum(1)
        inter_acc[~intersection] = 0 # set negative one to zeros

        reciproc = torch.div(inter_acc.float(), pred_acc.float()) # for each positive sample, divided by its rank index
        Map = torch.div(reciproc.sum(1), inter_sum1.float()) # normalized by positive numbers in each sample

        return ground_size, hit_size, Map.sum().item(), k*batch_size        
            

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--dataset_name', type=str, default='mimic',
                        help='name of the dataset')
    parser.add_argument('--data_path', type=str, default='',
                        help='path of the data')
    parser.add_argument('--image_embeddings_path', type=str, default='test_mimic_embeddings.pt',
                        help='image embeddings path')
    parser.add_argument('--sentence_embeddings_path', type=str, default='sentence_counts.pkl',
                        help='sentence embeddings path')
    parser.add_argument('--train_image_file_list', type=str, default='./data/mimic2/full_train_split.json',
                        help='the train image array')
    parser.add_argument('--dev_image_file_list', type=str, default='./data/mimic2/full_dev_split.json',
                        help='the dev image array')
    parser.add_argument('--test_image_file_list', type=str, default='./data/mimic2/full_test_split.json',
                        help='the test image array')
    parser.add_argument('--num_clusters', type=int, default=500)
    parser.add_argument('--num_negatives', type=int, default=4, help='Number of negatives')

    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./rank_pair_models/',
                        help='path for saving trained models')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='The path of loaded model')

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--transform_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--use_relu', action='store_true')
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    
    parser.add_argument('--use_center', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_sample', type=float, default=0)

    parser.add_argument('--seed', type=int, default=2019, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    # Handle CUDA things
    multi_gpus = True
    n_gpus = 0
    vis_devices_os = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not vis_devices_os:
        gpus = torch.cuda.device_count()
        n_gpus = gpus
        print('\n\n\n===========')
        print('No $CUDA_VISIBLE_DEVICES set, defaulting to {:,}'.format(gpus))
        print('===========\n\n\n')
        if gpus < 2:
            multi_gpus = False
        time.sleep(5)
    else:
        gpus = list(map(int, vis_devices_os.split(',')))
        n_gpus = len(gpus)
        if len(gpus) < 2:
            multi_gpus = False
            gpus = 1
        print('Visible devices as specified in $CUDA_VISIBLE_DEVICES: {}'.format(gpus))
    args.n_gpus = n_gpus
    
    # Reproducibility
    set_seed(args.seed, gpu=(n_gpus > 0))
    
    print(args)
    
    debugger = RankDebugger(args)
    # after fp16
    debugger.model = nn.DataParallel(debugger.model)
    
    debugger.train()
