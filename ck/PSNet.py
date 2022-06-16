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
    return label


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
            print(img_path)
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

dataset = ouluSingleLoader(list(range(1,999)))
print('# size of the current (sub)dataset is %d' % len(dataset))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                                 num_workers=4)

for i, batch in enumerate(dataloader):
    j = 0

exit()

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
parser.add_argument('--lr', type=float, default=0.0001,
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
load_encoderI = dirRoot + '/code/DTNet1/checkpoints/idnet_rafd/model/encoder_epoch_50.pth'
load_decoderI = dirRoot + '/code/DTNet1/checkpoints/idnet_rafd/model/decoder_epoch_50.pth'
load_encoderF = dirRoot + '/code/STVAE/checkpoints/pretrain_final/model/encoder_epoch_4.pth'
load_decoderF = dirRoot + '/code/STVAE/checkpoints/pretrain_final/model/decoder_epoch_4.pth'
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

descriptor = _VGG(ngpu)
encoderI = _Encoder(ngpu)
encoderI.apply(weights_init)
decoderI = _Decoder(ngpu)
decoderI.apply(weights_init)
encoderF = _Encoder(ngpu)
encoderF.apply(weights_init)
decoderF = _Decoder(ngpu)
decoderF.apply(weights_init)
# decoder = _Decoder(ngpu)
# decoder .apply(weights_init)

if args.encoder != '':
    print('LOAD MODEL!!!!!!!!!!!!!!!!!!!!!!')
    encoderI.load_state_dict(torch.load(load_encoderI, map_location='cpu'))
    decoderI.load_state_dict(torch.load(load_decoderI, map_location='cpu'))
    encoderF.load_state_dict(torch.load(load_encoderF, map_location='cpu'))
    decoderF.load_state_dict(torch.load(load_decoderF, map_location='cpu'))

if args.cuda:
    encoderI = encoderI.cuda()
    decoderI = decoderI.cuda()
    encoderF = encoderF.cuda()
    decoderF = decoderF.cuda()
    descriptor = descriptor.cuda()

mse = nn.MSELoss()
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        # print('gene_feature:', f.dtype)
        # print('target_feature:', f.dtype)
        fpl += mse(f, target.detach()).div(2.0)#.div(f.size(1))
    return fpl

def transform_convert(img_tensor, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return img_tensor

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

test_subject_str = args.test_subject
train_subject_str = args.train_subject
train_subject = []
test_subject = []
for sub in train_subject_str.split('_')[:-1]:
    train_subject.append(int(sub))
for sub in test_subject_str.split('_')[:-1]:
    test_subject.append(int(sub))
print('train_subject:', train_subject)
print('test_subject:', test_subject)

# TODO:训练ident

batch_size = 64
parameters1 = list(encoderI.parameters()) + list(decoderI.parameters())
optimizer1 = optim.Adam(parameters1, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-2)
alpha = 1.0
beta = 0.2
def train_idnet(k):
    iteration_count = 0
    for epoch in range(4):
        dataset = ouluPairLoaderAlign(train_subject)
        print('# size of the current (sub)dataset is %d' % len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=4)
        encoderI.train()
        decoderI.train()
        for i, batch in enumerate(dataloader):
            iteration_count += 1
            optimizer1.zero_grad()

            source = batch[0].cuda()
            target = batch[1].cuda()
            z, mu, logvar = encoderI(source)
            target_feature = descriptor(target)
            gene = decoderI(z)
            gene_feature = descriptor(gene)
            fpl_loss = fpl_criterion(gene_feature, target_feature)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = alpha * fpl_loss + beta * kld_loss

            loss.backward()
            optimizer1.step()
            _gene = transform_convert(gene)
            _source = transform_convert(source)
            _target = transform_convert(target)

            if iteration_count % args.log_interval == 0:
                print('[{}/{}][{}/{}] KLD: {:.4f} FPL: {:.4f} TOTAL: {:.4f}'.format(
                    epoch, args.nepoch, i, len(dataloader),
                    kld_loss, fpl_loss, loss))

            if iteration_count % args.img_interval == 0:
                visualizeAsImages(_source.data.clone(),
                                  dirImageoutput,
                                  filename='iter_' + str(iteration_count) + '_source_', n_sample=64, nrow=8,
                                  normalize=False)
                visualizeAsImages(_gene.data.clone(),
                                  dirImageoutput,
                                  filename='iter_' + str(iteration_count) + '_gene_', n_sample=64, nrow=8,
                                  normalize=False)
                visualizeAsImages(_target.data.clone(),
                                  dirImageoutput,
                                  filename='iter_' + str(iteration_count) + '_target_', n_sample=64, nrow=8,
                                  normalize=False)

        torch.save(encoderI.state_dict(), '{}/encoderI_epoch_{}.pth'.format(dirModel, epoch))
        torch.save(decoderI.state_dict(), '{}/decoderI_epoch_{}.pth'.format(dirModel, epoch))
        test_idnet(iteration_count)

def test_idnet(iteration_count):
    dataset = ouluPairLoaderAlignTest(test_subject)
    print('# size of the current (sub)dataset is %d' % len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4)
    encoderI.eval()
    decoderI.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            source = batch[0].cuda()
            target = batch[1].cuda()
            z, mu, logvar = encoderI(source)
            target_feature = descriptor(target)
            gene = decoderI(z)
            gene_feature = descriptor(gene)
            fpl_loss = fpl_criterion(gene_feature, target_feature)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = alpha * fpl_loss + beta * kld_loss

        _gene = transform_convert(gene)
        _source = transform_convert(source)
        _target = transform_convert(target)

        if iteration_count % 1 == 0:
            print('[{}/{}] KLD: {:.4f} FPL: {:.4f} TOTAL: {:.4f}'.format(
                 i, len(dataloader),
                kld_loss, fpl_loss, loss))

        visualizeAsImages(_source.data.clone(),
                              dirTestingoutput,
                              filename='iter_' + str(iteration_count) + '_source_', n_sample=64, nrow=8,
                              normalize=False)
        visualizeAsImages(_gene.data.clone(),
                              dirTestingoutput,
                              filename='iter_' + str(iteration_count) + '_gene_', n_sample=64, nrow=8,
                              normalize=False)
        visualizeAsImages(_target.data.clone(),
                          dirTestingoutput,
                          filename='iter_' + str(iteration_count) + '_target_', n_sample=64, nrow=8,
                          normalize=False)
    encoderI.train()
    decoderI.train()

train_idnet(-1)


batch_size = 64
parameters3 = list(encoderF.parameters()) + list(decoderF.parameters())
optimizer3 = optim.Adam(parameters3, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-2)
def train_facenet():
    iteration_count = 0
    for epoch in range(4):
        dataset = ouluSingleLoader(train_subject)
        print('# size of the current (sub)dataset is %d' % len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=4)
        encoderF.train()
        decoderF.train()
        for i, batch in enumerate(dataloader):
            iteration_count += 1
            optimizer3.zero_grad()

            source = batch[0].cuda()
            z, mu, logvar = encoderF(source)
            target_feature = descriptor(source)
            gene = decoderF(z)
            gene_feature = descriptor(gene)
            fpl_loss = fpl_criterion(gene_feature, target_feature)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = alpha * fpl_loss + beta * kld_loss

            loss.backward()
            optimizer3.step()
            _gene = transform_convert(gene)
            _source = transform_convert(source)

            if iteration_count % args.log_interval == 0:
                print('[{}/{}][{}/{}] KLD: {:.4f} FPL: {:.4f} TOTAL: {:.4f}'.format(
                    epoch, args.nepoch, i, len(dataloader),
                    kld_loss, fpl_loss, loss))


            if iteration_count % args.img_interval == 0:
                visualizeAsImages(_source.data.clone(),
                                  dirImageoutput,
                                  filename='iter_' + str(iteration_count) + '_source_', n_sample=64, nrow=8,
                                  normalize=False)
                visualizeAsImages(_gene.data.clone(),
                                  dirImageoutput,
                                  filename='iter_' + str(iteration_count) + '_gene_', n_sample=64, nrow=8,
                                  normalize=False)
        # test_idnet(iteration_count)
        torch.save(encoderF.state_dict(), '{}/encoder_epoch_{}.pth'.format(dirModel, epoch))
        torch.save(decoderF.state_dict(), '{}/decoder_epoch_{}.pth'.format(dirModel, epoch))

train_facenet()


encoderI.eval()



batch_size = 64
extractor = nn.Sequential(
    nn.Linear(200, 100)
    # nn.ReLU(),
    # nn.Linear(32, 6)
)
classifier = nn.Sequential(
    torch.nn.Dropout (p= 0.5, inplace= False),
    nn.Linear(100, 7)
    # nn.ReLU(),
    # nn.Linear(32, 6)
)
beta = 0.5
classifier.apply(weights_init)
classifier = classifier.cuda()
extractor.apply(weights_init)
extractor = extractor.cuda()
parameters2 = list(encoderF.parameters()) + list(classifier.parameters()) + list(extractor.parameters())
# list(encoder.parameters()) + list(decoder.parameters())
optimizer2 = optim.Adam(parameters2, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-2)
criterion = nn.CrossEntropyLoss().cuda()
def train_classifier(k):
    iteration_count = 0
    bestacc = 0
    best_labels = []
    best_preds = []
    for epoch in range(10):
        dataset = ouluSingleLoader(train_subject)
        # print('# size of the current (sub)dataset is %d' % len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=4)
        classifier.train()
        for i, batch in enumerate(dataloader):
            iteration_count += 1
            optimizer2.zero_grad()

            img = batch[0].cuda()
            label = batch[1].cuda()
            path = batch[2]
            _, imgI, _ = encoderI(img)
            _, imgF, _ = encoderF(img)
            residue = imgF - imgI
            feature = torch.cat((imgI, imgF), 1)
            featureE = extractor(feature)

            out = classifier(featureE)
            loss1 = criterion(out, label)
            loss2 = mse(residue, featureE)

            loss = loss1 + beta * loss2
            loss.backward()
            optimizer2.step()

            if iteration_count % args.log_interval == 0:
                print('[{}/{}][{}/{}] CLS: {:.4f} LOSS2: {:.4f} LOSS: {:.4f} Acc: {}'.format(
                    epoch, args.nepoch, i, len(dataloader), loss1, loss2, loss, accuracyComp(out, label)))
            if i % 500 == 0:
                acc, y_labels, y_preds = test_classifier(iteration_count, 0)
                if acc > bestacc:
                    bestacc = acc
                    best_labels = y_labels
                    best_preds = y_preds
                    # print('bestacc is:', bestacc)
        torch.save(encoderF.state_dict(), '{}/encoderF_epoch_{}.pth'.format(dirModel, epoch))
        torch.save(classifier.state_dict(), '{}/classifier_epoch_{}.pth'.format(dirModel, epoch))
        torch.save(extractor.state_dict(), '{}/extractor_epoch_{}.pth'.format(dirModel, epoch))
    return bestacc, best_labels, best_preds

def correctCount(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    #     print(pred.shape(),label.shape())

    test_np = (np.argmax(pred, 1) == label)
    correct = 0
    for t in test_np:
        if t == True:
            correct += 1
    return correct


def accuracyComp(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    #     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

def test_classifier(iteration_count, k):
    dataset = ouluSingleTestLoader(test_subject)
    # print('# size of the current (sub)dataset is %d' % len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4)
    encoderF.eval()
    classifier.eval()
    extractor.eval()
    correct = 0
    y_labels = []
    y_preds = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            img = batch[0].cuda()
            label = batch[1].cuda()
            path = batch[2]
            _, imgI, _ = encoderI(img)
            _, imgF, _ = encoderF(img)
            residue = imgF - imgI
            feature = torch.cat((imgI, imgF), 1)
            featureE = extractor(feature)

            out = classifier(featureE)
            loss1 = criterion(out, label)
            loss2 = mse(residue, featureE)
            loss = loss1 + beta * loss2

            correct += correctCount(out, label)

            y_pred = out.cpu().data.numpy()
            y_label = label.cpu().data.numpy()
            for p in np.argmax(y_pred, 1):
                y_preds.append(p)
            for q in y_label:
                y_labels.append(q)

            print('[{}/{}] LOSS1: {:.4f} LOSS2: {:.4f} LOSS: {:.4f} Acc: {}'.format(
                i, len(dataloader), loss1, loss2, loss, accuracyComp(out, label)))
    # print('accuracy in testset is: ', correct / len(dataset))
    acc = correct / len(dataset)
    classifier.train()
    encoderF.train()
    extractor.train()
    return acc, y_labels, y_preds

# bestaccs = []
# for k in range(10):
#     if args.encoder != '':
#         print('LOAD MODEL!!!!!!!!!!!!!!!!!!!!!!')
#         encoderI.load_state_dict(torch.load(load_encoderI, map_location='cpu'))
#         encoderF.load_state_dict(torch.load(load_encoderF, map_location='cpu'))
#
#     if args.cuda:
#         encoderI = encoderI.cuda()
#         encoderF = encoderF.cuda()
#     bestaccs.append(train_classifier(k))
# print('bestacc is:', bestaccs)
# print('sum is:', sum(bestaccs))
acc, labels, preds = train_classifier(-1)
print('bestacc is:', acc)
print('labels:', labels)
print('preds:', preds)

label_path = dirCheckpoints + '/label.txt'
pred_path = dirCheckpoints + '/pred.txt'

with open(label_path, 'a') as f:
    f.write(str(labels))
    f.write('\n')

with open(pred_path, 'a') as f:
    f.write(str(preds))
    f.write('\n')
