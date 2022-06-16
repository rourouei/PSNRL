import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import argparse
import torchvision.transforms as transforms
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
# 设置使用的GPU
torch.cuda.set_device(0)

# TODO: 定义oulu dataloader ------------------------------------------------------------------
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


data_dir = '/home/njuciairs/zmy/data'
oulu = data_dir + '/ck_aug'
oulu_dirs = []
for path in os.listdir(oulu):
    oulu_dirs.append(oulu+'/'+path)
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
mytransform = transforms.Compose([
    #transforms.Scale(64),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6，13，17，34，62，66是不存在的
class ouluPairLoaderAlign(data.Dataset):
    def __init__(self, subject):
        super(ouluPairLoaderAlign, self).__init__()
        pairs = []
        for dir in oulu_dirs:
            for i in subject:
                if i < 100:
                    dirpath = dir + '/S0' + str(i)
                    # (dirpath)
                else:
                    dirpath = dir + '/S' + str(i)
                subjects = []
                for root, _, fnames in os.walk(dirpath):
                    apex = []
                    onset = []
                    fnames.sort()
                    for file in fnames[-3:]:
                        apex.append(os.path.join(root, file))
                    for file in fnames[:1]:
                        onset.append(os.path.join(root, file))
                    for a in apex:
                        for o in onset:
                            pair = [a, o]
                            pairs.append(pair)
        self.pairs = pairs

    def __getitem__(self, index):
        pair = self.pairs[index]
        # print(pair)
        exprPath = pair[0]
        neuPath = pair[1]
        expr = Image.open(exprPath).convert('RGB')
        expr = mytransform(expr)
        neu = Image.open(neuPath).convert('RGB')
        neu = mytransform(neu)
        return expr, neu, exprPath, neuPath

    def __len__(self):
        return len(self.pairs)

class ouluPairLoaderAlignTest(data.Dataset):
    def __init__(self, subject):
        super(ouluPairLoaderAlignTest, self).__init__()
        pairs = []
        oulu_dirs = [oulu + '/ck_center']
        for dir in oulu_dirs:
            for i in subject:
                if i < 100:
                    dirpath = dir + '/S0' + str(i)
                    # (dirpath)
                else:
                    dirpath = dir + '/S' + str(i)
                subjects = []
                for root, _, fnames in os.walk(dirpath):
                    apex = []
                    onset = []
                    fnames.sort()
                    for file in fnames[-3:]:
                        apex.append(os.path.join(root, file))
                    for file in fnames[:1]:
                        onset.append(os.path.join(root, file))
                    for a in apex:
                        for o in onset:
                            pair = [a, o]
                            pairs.append(pair)
        self.pairs = pairs

    def __getitem__(self, index):
        pair = self.pairs[index]
        # print(pair)
        exprPath = pair[0]
        neuPath = pair[1]
        expr = Image.open(exprPath).convert('RGB')
        expr = mytransform(expr)
        neu = Image.open(neuPath).convert('RGB')
        neu = mytransform(neu)
        return expr, neu, exprPath, neuPath

    def __len__(self):
        return len(self.pairs)

# dataset = ouluPairLoaderAlign([i for i in range(1, 81)])
# print('# size of the current (sub)dataset is %d' % len(dataset))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
#                                              num_workers=4)
# for i, batch in enumerate(dataloader):
#     print(batch[2])
#     print(batch[3])
#     exit()
# exit()

def get_label(img_path):
    emo_dir = data_dir + '/Emotion/'
    label_dir = emo_dir + '/' + img_path.split('/')[-3]  + '/' + img_path.split('/')[-2]
    label = -1
    for root, _, files in os.walk(label_dir):
        if len(files) == 0:
            return -1
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                label_str = f.read()
            label = float(label_str)
    return int(label)


class ouluSingleLoader(data.Dataset):
    def __init__(self, subject, last=3):
        super(ouluSingleLoader, self).__init__()
        images = []
        labels = []
        for dir in oulu_dirs:
            for i in subject:
                if i < 100:
                    dirpath = dir + '/S0' + str(i)
                    # (dirpath)
                else:
                    dirpath = dir + '/S' + str(i)
                for root, _, fnames in os.walk(dirpath):
                    fnames.sort()
                    for fname in fnames[-last:]:
                        img = os.path.join(root, fname)
                        label = get_label(img)
                        if (label != -1):
                            images.append(img)
                            labels.append(label)
        self.images = images
        self.labels = labels
        # setl = set(labels)
        # # print(labels)
        # print(setl)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        img = mytransform(img)
        l = self.labels[index]
        if l == 1.0:
            label = 0
        elif l == 2.0:
            label = 1
        elif l == 3.0:
            label = 2
        elif l == 4.0:
            label = 3
        elif l == 5.0:
            label = 4
        elif l == 6.0:
            label = 5
        elif l == 7.0:
            label = 6
        else:
            print(img_path)
            print('l:', l)
        # print(img_path, ':', label)
        return img, label, img_path

    def __len__(self):
        return len(self.images)


class ouluSingleTestLoader(data.Dataset):
    def __init__(self, subject, last=3):
        super(ouluSingleTestLoader, self).__init__()
        images = []
        labels = []
        oulu_dirs = [data_dir + '/ck_aug/ck_center']
        for dir in oulu_dirs:
            for i in subject:
                if i < 100:
                    dirpath = dir + '/S0' + str(i)
                    # (dirpath)
                else:
                    dirpath = dir + '/S' + str(i)
                for root, _, fnames in os.walk(dirpath):
                    fnames.sort()
                    for fname in fnames[-last:]:
                        img = os.path.join(root, fname)
                        label = get_label(img)
                        if (label != -1):
                            images.append(img)
                            labels.append(label)
        self.images = images
        self.labels = labels
        # setl = set(labels)
        # # print(labels)
        # print(setl)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        img = mytransform(img)
        l = self.labels[index]
        if l == 1.0:
            label = 0
        elif l == 2.0:
            label = 1
        elif l == 3.0:
            label = 2
        elif l == 4.0:
            label = 3
        elif l == 5.0:
            label = 4
        elif l == 6.0:
            label = 5
        elif l == 7.0:
            label = 6
        else:
            print(img_path)
            print('l:', l)
        # print(label)
        return img, label, img_path

    def __len__(self):
        return len(self.images)

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
# default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']
default_content_layers = ['relu1_1', 'relu2_1', 'relu3_1']
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers, default=2', default=2)
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size, default=8')
parser.add_argument('--image_size', type=int, default=64,
                    help='height/width length of the input images, default=64')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent vector z, default=100')
parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance norm layer instead of batch norm')
parser.add_argument('--content_layers', type=str, nargs='?', default=None,
                    help='name of the layers to be used to compute the feature perceptual loss, default=[relu3_1, relu4_1, relu5_1]')
parser.add_argument('--nepoch', type=int, default=50,
                    help='number of epochs to train for, default=5')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--encoder', default='1',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manual_seed', type=int, default=1234, help='manual seed')
parser.add_argument('--log_interval', type=int, default=10, help='number of iterations between each stdout logging, default=1')
parser.add_argument('--img_interval', type=int, default=1000, help='number of iterations between each image saving, default=100')
parser.add_argument('--test_subject', type=str, nargs='?', default=None,
                    help='test_split[k]')
parser.add_argument('--train_subject', type=str, nargs='?', default=None,
                    help='train_split[k]')


args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
print("Random Seed: ", args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manual_seed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(args.ngpu)
nz = int(args.nz)
nef = int(args.nef)
ndf = int(args.ndf)
nc = 3
out_size = args.image_size // 16
if args.instance_norm:
    Normalize = nn.InstanceNorm2d
    print('instanse norm!!!!!!!!!!')
else:
    Normalize = nn.BatchNorm2d
    print('batch norm!!!!!!!!!!')
if args.content_layers is None:
    content_layers = default_content_layers
else:
    content_layers = args.content_layers

class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self, ngpu):
        super(_VGG, self).__init__()

        self.ngpu = ngpu
        features = models.vgg19(pretrained=True).features
        # print(len(features))
        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            # print(name)
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(
                    module, output, range(self.ngpu))
            else:
                output = module(output)
            if name in content_layers:
                #print('!!!', name)
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

class _Encoder(nn.Module):
    '''
    Encoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            Normalize(nef),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            Normalize(nef * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            Normalize(nef * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            Normalize(nef * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.encoder, input, range(self.ngpu))
            hidden = hidden.view(batch_size, -1)
            mean = nn.parallel.data_parallel(
                self.mean, hidden, range(self.ngpu))
            logvar = nn.parallel.data_parallel(
                self.logvar, hidden, range(self.ngpu))
        else:
            hidden = self.encoder(input)
            hidden = hidden.view(batch_size, -1)
            mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z, mean, logvar

class _Decoder(nn.Module):
    '''
    Decoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * out_size * out_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            Normalize(ndf * 4, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            Normalize(ndf * 2, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            Normalize(ndf, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.decoder_dense, input, range(self.ngpu))
            hidden = hidden.view(batch_size, ndf * 8, out_size, out_size)
            output = nn.parallel.data_parallel(
                self.decoder_conv, input, range(self.ngpu))
        else:
            hidden = self.decoder_dense(input).view(
                batch_size, ndf * 8, out_size, out_size)
            output = self.decoder_conv(hidden)
        return output

def weights_init(m):
    '''
    Custom weights initialization called on encoder and decoder.
    '''
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

import cv2
def visualizeAsImages(img_list, output_dir,
                      n_sample=4, id_sample=None, dim=-1,
                      filename='myimage', nrow=2,
                      normalize=True):
    if id_sample is None:
        images = img_list[0:n_sample, :, :, :]
    else:
        images = img_list[id_sample, :, :, :]
    if dim >= 0:
        images = images[:, dim, :, :].unsqueeze(1)
    vutils.save_image(images,
                      '%s/%s' % (output_dir, filename + '.png'),
                      nrow=nrow, normalize=normalize, padding=2)

dirRoot = '/home/njuciairs/zmy'
dirCheckpoints = dirRoot + '/code/PSNRL/checkpoints/psnet_ck'
dirImageoutput = dirCheckpoints + '/ck_train'
dirTestingoutput = dirCheckpoints + '/ck_test'
dirModel = dirCheckpoints + '/model'
# ident 50, facenet 10
load_encoderI = dirRoot + '/code/PSNRL/checkpoints/psnet_ck/model/encoderI_epoch_3.pth'
# load_decoderI = dirRoot + '/code/PSNRL/checkpoints/psnet_ck/model/decoder_epoch_4.pth'
load_encoderF = dirRoot + '/code/PSNRL/checkpoints/psnet_ck/model/encoderF_epoch_9.pth'
load_extractor = dirRoot + '/code/PSNRL/checkpoints/psnet_ck/model/extractor_epoch_9.pth'
# load_decoderF = dirRoot + '/code/PSNRL/checkpoints/psnet_ck/model/decoderI_epoch_4.pth'
# load_decoder = dirRoot + '/code/DTNet1/checkpoints/facenet_rafd/model/decoder_epoch_10.pth'
dir_ran = dirCheckpoints + '/random_gen'
try:
    os.makedirs(dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(dirTestingoutput)
except OSError:
    pass
try:
    os.makedirs(dirModel)
except OSError:
    pass
try:
    os.makedirs(dir_ran)
except OSError:
    pass


encoderI = _Encoder(ngpu)
encoderI.apply(weights_init)
encoderF = _Encoder(ngpu)
encoderF.apply(weights_init)
extractor = nn.Sequential(
    nn.Linear(200, 100)
    # nn.ReLU(),
    # nn.Linear(32, 6)
)

if args.encoder != '':
    print('LOAD MODEL!!!!!!!!!!!!!!!!!!!!!!')
    encoderI.load_state_dict(torch.load(load_encoderI, map_location='cpu'))
    encoderF.load_state_dict(torch.load(load_encoderF, map_location='cpu'))
    extractor.load_state_dict(torch.load(load_extractor, map_location='cpu'))

if args.cuda:
    encoderI = encoderI.cuda()
    encoderF = encoderF.cuda()
    extractor = extractor.cuda()

encoderI.eval()
encoderF.eval()
extractor.eval()

from sklearn.model_selection import  KFold

FOLD = 10
kfold = KFold(FOLD, shuffle=True, random_state=None)
train_split = []
test_split = []

subject = []
strs = ['S087', 'S138', 'S005', 'S067', 'S063', 'S035', 'S096', 'S084', 'S032', 'S074', 'S056', 'S147', 'S504', 'S037', 'S108', 'S125', 'S109', 'S133', 'S099', 'S112', 'S160', 'S011', 'S502', 'S155', 'S065', 'S071', 'S135', 'S122', 'S999', 'S044', 'S100', 'S026', 'S129', 'S154', 'S080', 'S092', 'S115', 'S091', 'S895', 'S070', 'S029', 'S103', 'S086', 'S132', 'S148', 'S137', 'S068', 'S121', 'S076', 'S127', 'S130', 'S069', 'S053', 'S131', 'S107', 'S139', 'S072', 'S505', 'S506', 'S058', 'S083', 'S095', 'S111', 'S057', 'S077', 'S149', 'S114', 'S098', 'S075', 'S128', 'S116', 'S042', 'S078', 'S082', 'S093', 'S120', 'S061', 'S066', 'S106', 'S158', 'S010', 'S151', 'S110', 'S085', 'S022', 'S054', 'S119', 'S046', 'S157', 'S028', 'S104', 'S081', 'S055', 'S045', 'S090', 'S088', 'S079', 'S097', 'S060', 'S052', 'S503', 'S126', 'S073', 'S136', 'S117', 'S156', 'S105', 'S014', 'S102', 'S059', 'S064', 'S051', 'S118', 'S113', 'S062', 'S094', 'S134', 'S124', 'S034', 'S101', 'S501', 'S089', 'S050']

for s in strs:
    subject.append(int(s[1:]))


for i, (train_index, test_index) in enumerate(kfold.split(subject)):
    #print('Fold: ', i)
    train_subjects = [subject[i] for i in train_index]
    test_subjects = [subject[i] for i in test_index]
    train_split.append(train_subjects)
    test_split.append(test_subjects)

# residue = [torch.zeros((100)),torch.zeros((100)),torch.zeros((100)),torch.zeros((100)),torch.zeros((100)),torch.zeros((100)),torch.zeros((100))]
# print(len(residue))
# print(residue[4])
# count = [0,0,0,0,0,0,0]
# oulu_dirs = [data_dir + '/ck_aug/ck_center']
# all_subject = []
# for sub in train_split[0]:
#     all_subject.append(sub)
# for sub in test_split[0]:
#     all_subject.append(sub)


# for dir in oulu_dirs:
#     for i in subject:
#         if i < 100:
#             dirpath = dir + '/S0' + str(i)
#             # (dirpath)
#         else:
#             dirpath = dir + '/S' + str(i)
#         for root, _, fnames in os.walk(dirpath):
#             fnames.sort()
#             for fname in fnames[-3:]:
#                 path = os.path.join(root, fname)
#                 label = get_label(path)
#                 if label != -1:
#                     label -= 1
#                 else:
#                     continue
#                 img = Image.open(path).convert('RGB')
#                 img = mytransform(img).cuda()
#                 img = torch.unsqueeze(img, dim=0)
#                 _, imgI, _ = encoderI(img)
#                 _, imgF, _ = encoderF(img)
#                 #residue = imgF - imgI
#                 feature = torch.cat((imgI, imgF), 1)
#                 featureE = extractor(feature)
#                 featureE = featureE.squeeze(dim=0)
#                 # print(featureE.shape)
#                 # print(residue[1].shape)
#                 # print(label)
#                 residue[label] = residue[label] + featureE.detach().cpu()
#                 count[label] += 1


# for dir in oulu_dirs:
#     for i in subject:
#         if i < 100:
#             dirpath = dir + '/S0' + str(i)
#             # (dirpath)
#         else:
#             dirpath = dir + '/S' + str(i)
#         for root, _, fnames in os.walk(dirpath):
#             fnames.sort()
#             for fname in fnames[-3:]:
#                 path = os.path.join(root, fname)
#                 label = get_label(path)
#                 if label != -1:
#                     label -= 1
#                 else:
#                     continue
#                 img = Image.open(path).convert('RGB')
#                 img = mytransform(img).cuda()
#                 img = torch.unsqueeze(img, dim=0)
#                 _, imgI, _ = encoderI(img)
#                 _, imgF, _ = encoderF(img)
#                 #residue = imgF - imgI
#                 # feature = torch.cat((imgI, imgF), 1)
#                 # featureE = extractor(feature)
#                 # featureE = featureE.squeeze(dim=0)
#                 # print(featureE.shape)
#                 # print(residue[1].shape)
#                 # print(label)
#                 residue[label] = residue[label] + imgF.detach().cpu()
#                 count[label] += 1
#
# for i in range(0, 7):
#     residue[i] = residue[i]/count[i]
#
# res_path = dirCheckpoints + '/f.txt'
#
#
# with open(res_path, 'a') as f:
#     for i in range(0, 7):
#         f.write(str(residue[i]))
#         f.write('\n')

expr = ['0', '1', '2', '3', '4', '5','6']
for e in expr:
    dirOutput = '/home/njuciairs/zmy/code/PSNRL/output/' + e
    try:
        os.makedirs(dirOutput)
    except OSError:
        pass

def get_label(img_path):
    emo_dir = data_dir + '/Emotion/'
    label_dir = emo_dir + '/' + img_path.split('/')[-3]  + '/' + img_path.split('/')[-2]
    label = -1
    for root, _, files in os.walk(label_dir):
        if len(files) == 0:
            return -1
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                label_str = f.read()
            label = float(label_str)
    return int(label)

# a = [21, 22, 35, 36, 40, 68, 71, 72]
a = list(range(1, 1000))
def get_z(expr):
    for rafd in oulu_dirs:
        for i in a:
            if i < 100:
                dirpath = rafd + '/S0' + str(i)
                # (dirpath)
            else:
                dirpath = rafd + '/S' + str(i)

            for root, _, fnames in os.walk(dirpath):
                if len(fnames) == 0:
                    continue
                for file in sorted(fnames)[-3:]:
                    path = os.path.join(root, file)
                    label = get_label(path)
                    if label != -1:
                        label -= 1
                    else:
                        continue
                    print(label)
                    img = Image.open(path).convert('RGB')
                    img = mytransform(img).cuda()
                    img = torch.unsqueeze(img, dim=0)

                    _, imgI, _ = encoderI(img)
                    _, imgF, _ = encoderF(img)
                                    #residue = imgF - imgI
                    feature = torch.cat((imgI, imgF), 1)
                    featureE = extractor(feature)
                    arr = imgF.detach().cpu().numpy()
                    txt_path ='/home/njuciairs/zmy/code/PSNRL/output/' + str(label) + '/' + str(i) + '_' + file[:-5]
                    np.savetxt(txt_path + '.txt', arr)

get_z(expr)








