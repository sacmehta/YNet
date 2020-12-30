#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the code for training and validation
# ==============================================================================
import loadData as ld
import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import VisualizeGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import numpy as np


def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.data[0])

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def train(args, train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data[0])
        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)


def trainValidateSegmentation(args):
    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoader = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        if dataLoader is None:
            print('Error while processing the data. Please check')
            exit(-1)
        data = dataLoader.processData()
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    if args.modelType == 'C1':
        model = net.ResNetC1(args.classes)
    elif args.modelType == 'D1':
        model = net.ResNetD1(args.classes)
    else:
        print('Please select the correct model. Exiting!!')
        exit(-1)

        args.savedir = args.savedir + args.modelType + '/'

    if args.onGPU == True:
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.onGPU == True:
        model = model.cuda()

    if args.visualizeNet == True:
        x = Variable(torch.randn(1, 3, args.inWidth, args.inHeight))

        if args.onGPU == True:
            x = x.cuda()

        y = model.forward(x)
        g = viz.make_dot(y)
        g.render(args.savedir + '/model.png', view=False)

    n_param = sum([np.prod(param.size()) for param in model.parameters()])
    print('Network parameters: ' + str(n_param))

    # define optimization criteria
    print('Weights to handle class-imbalance')
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    print(weight)
    if args.onGPU == True:
        weight = weight.cuda()

    criteria = CrossEntropyLoss2d(weight)  # weight

    if args.onGPU == True:
        criteria = criteria.cuda()

    trainDatasetNoZoom = myTransforms.Compose([
        # myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.RandomCropResize(20),
        myTransforms.RandomHorizontalFlip(),
        myTransforms.ToTensor(args.scaleIn)
    ])

    trainDatasetWithZoom = myTransforms.Compose([
        # myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Zoom(512, 512),
        myTransforms.RandomCropResize(20),
        myTransforms.RandomHorizontalFlip(),
        myTransforms.ToTensor(args.scaleIn)
    ])

    valDataset = myTransforms.Compose([
        # myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.ToTensor(args.scaleIn)
    ])

    trainLoaderNoZoom = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetNoZoom),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainLoaderWithZoom = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetWithZoom),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.onGPU == True:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resumeLoc))
            checkpoint = torch.load(args.resumeLoc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + os.sep + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("Parameters: %s" % (str(n_param)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(n_param)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
        logger.flush()

    #lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.1)

    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step(epoch)

        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # run at zoomed images first
        train(args, trainLoaderWithZoom, model, criteria, optimizer, epoch)
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(args, trainLoaderNoZoom, model,
                                                                                   criteria, optimizer, epoch)
        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'iouTr': mIOU_tr,
            'iouVal': mIOU_val,
        }, args.savedir + '/checkpoint.pth.tar')

        # save the model also
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), model_file_name)

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write(
                "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (
        epoch, lossTr, lossVal, mIOU_tr, mIOU_val))

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Training YNet')
    parser.add_argument('--model', default="YNet", help='Name of the network')
    parser.add_argument('--data_dir', default="./data/", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=384, help='Width of the input patch')
    parser.add_argument('--inHeight', type=int, default=384, help='Height of the input patch')
    parser.add_argument('--scaleIn', type=int, default=1, help='scaling factor for training the models at '
                                                               'low resolution first and then full resolution.'
                                                               'We did not use it.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for processing the data')
    parser.add_argument('--batch_size', type=int, default=10, help='batch ize')
    parser.add_argument('--step_loss', type=int, default=100, help='decay the learning rate after these many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--savedir', default='./results', help='results directory')
    parser.add_argument('--visualizeNet', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False, help='Use this flag to load the last checkpoint for training')
    parser.add_argument('--resumeLoc', default='./results_C1/checkpoint.pth.tar', help='checkpoint location')
    parser.add_argument('--classes', type=int, default=8, help='Number of classes in the dataset')
    parser.add_argument('--cached_data_file', default='ynet_cache.p', help='Data file names and other values, such as'
                                                                           'class weights, are cached')
    parser.add_argument('--logFile', default='trainValLog.txt', help="Log file")
    parser.add_argument('--onGPU', default=True, help='True if you want to train on GPU')
    parser.add_argument('--modelType', default='C1', help='Model could be C1 or D1')

    args = parser.parse_args()
    assert args.modelType in ['C1', 'D1']
    args.savedir = args.savedir + '_' + args.modelType + os.sep # update the save dir name with model type
    trainValidateSegmentation(args)
