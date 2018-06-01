import loadData as ld
import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import VisualizeGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim.lr_scheduler
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval

def val(args, val_loader, model, criterion, criterion1):
    #switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)
    iouDiagEvalVal = iouEval(args.diagClasses)

    epoch_loss = []
    class_loss = []

    total_batches = len(val_loader)
    for i, (input, target, target2) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
            target2 = target2.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target2_var = torch.autograd.Variable(target2)

        # run the mdoel
        output, output1 = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)
        loss1 = criterion1(output1, target2_var)

        epoch_loss.append(loss.data[0])
        class_loss.append(loss1.data[0])

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)
        iouDiagEvalVal.addBatch(output1.max(1)[1].data, target2_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    average_epoch_class_loss = sum(class_loss) / len(class_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()
    overall_acc1, per_class_acc1, per_class_iu1, mIOU1 = iouDiagEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU, average_epoch_class_loss, overall_acc1, per_class_acc1, per_class_iu1, mIOU1


def train(args, train_loader, model, criterion, criterion1, optimizer, epoch):

    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(args.classes)
    iouDiagEvalTrain = iouEval(args.diagClasses)

    epoch_loss = []
    class_loss = []

    total_batches = len(train_loader)
    for i, (input, target, target2) in enumerate(train_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
            target2 = target2.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target2_var = torch.autograd.Variable(target2)

        #run the mdoel
        output, output1 = model(input_var)

        #set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)
        loss1 = criterion1(output1, target2_var)

        optimizer.zero_grad()
        loss1.backward(retain_graph=True) # you need to keep the graph from classification branch so that it can be used
                                            # during the update from the segmentation branch
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data[0])
        class_loss.append(loss1.data[0])
        time_taken = time.time() - start_time

        #compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)
        iouDiagEvalTrain.addBatch(output1.max(1)[1].data, target2_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    average_epoch_class_loss = sum(class_loss) / len(class_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()
    overall_acc1, per_class_acc1, per_class_iu1, mIOU1 = iouDiagEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU, average_epoch_class_loss, overall_acc1, per_class_acc1, per_class_iu1, mIOU1

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def trainValidateSegmentation(args):

    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoader = ld.LoadData(args.data_dir, args.classes, args.diagClasses, args.cached_data_file)
        if dataLoader == None:
            print('Error while caching the data. Please check')
            exit(-1)
        data = dataLoader.processData()
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    if args.modelType == 'C1':
        model = net.ResNetC1_YNet(args.classes, args.diagClasses, args.pretrainedSeg)
    elif args.modelType == 'D1':
        model = net.ResNetD1_YNet(args.classes, args.diagClasses, args.pretrainedSeg)
    else:
        print('Please select the correct model. Exiting!!')
        exit(-1)

    if args.onGPU == True:
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.visualizeNet == True:
        x = Variable(torch.randn(1, 3, args.inWidth, args.inHeight))

        if args.onGPU == True:
            x = x.cuda()

        y, y1 = model.forward(x)
        g = viz.make_dot(y)
        g1 = viz.make_dot(y1)
        g.render(args.savedir + '/model_seg.png', view=False)
        g1.render(args.savedir + '/model_class.png', view=False)
    n_param = sum([np.prod(param.size()) for param in model.parameters()])
    print('Network parameters: ' + str(n_param))

    # define optimization criteria
    print('Weights to handle class-imbalance')
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    print(weight)
    criteria = CrossEntropyLoss2d(weight)
    if args.onGPU == True:
        weight = weight.cuda()

    #uncomment if you want to use it on diagnostic labels
    criteria1 = torch.nn.CrossEntropyLoss()
    #weightDiag = torch.from_numpy(data['diagClassWeights'])
    #print(weightDiag)
    #if args.onGPU == True:
    #    weightDiag = weightDiag.cuda()
    # criteria1 = torch.nn.CrossEntropyLoss(weightDiag)

    if args.onGPU == True:
        criteria = criteria.cuda()
        criteria1 = criteria1.cuda()

    trainDataset = myTransforms.Compose([
            myTransforms.RandomCropResize(20),
            myTransforms.RandomHorizontalFlip(),
            #myTransforms.RandomCrop(64),
            #myTransforms.Normalize(mean=data['mean'], std=data['std']),
            myTransforms.ToTensor(args.scaleIn),
            #
        ])
    
    trainDataset3 = myTransforms.Compose([
        myTransforms.Zoom(512, 512),
        myTransforms.RandomCropResize(20),
        myTransforms.RandomHorizontalFlip(),
        # myTransforms.RandomCrop(64),
        # myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    valDataset = myTransforms.Compose([
            #myTransforms.Normalize(mean=data['mean'], std=data['std']),
            myTransforms.ToTensor(args.scaleIn),
            #
        ])

    trainLoader = torch.utils.data.DataLoader(
                        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], data['trainDiag'], transform=trainDataset),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
    trainLoader3 = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], data['trainDiag'], transform=trainDataset3),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
                myDataLoader.MyDataset(data['valIm'], data['valAnnot'], data['valDiag'], transform=valDataset),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # define the optimizer
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4)

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

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(n_param)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))

	
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.1)

    for epoch in range(start_epoch, args.max_epochs):
        scheduler.step(epoch)

        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        # train for one epoch
        # run at low resolution first
        train(args, trainLoader3, model, criteria, criteria1, optimizer, epoch)
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr, lossTr1, overall_acc_tr1, per_class_acc_tr1, per_class_iu_tr1, mIOU_tr1 = train(args, trainLoader, model, criteria, criteria1, optimizer, epoch)
        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val, lossVal1, overall_acc_val1, per_class_acc_val1, per_class_iu_val1, mIOU_val1 = val(args, valLoader, model, criteria, criteria1)

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

        #save the model also
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), model_file_name)

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))
            log.write('Classification Results')
            log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                epoch, overall_acc_tr1, overall_acc_val1, mIOU_tr1, mIOU_val1))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr1))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val1))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr1))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val1))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))

    logger.close()



if __name__ == '__main__':

    parser = ArgumentParser(description='Training YNet (Jointly)')
    parser.add_argument('--model', default="ynet", help='Name of the network')
    parser.add_argument('--data_dir', default="../stage1/data/")
    parser.add_argument('--inWidth', type=int, default=384, help='Width of the input patch')
    parser.add_argument('--inHeight', type=int, default=384, help='Height of the input patch')
    parser.add_argument('--scaleIn', type=int, default=1, help='scaling factor for training the models at '
                                                               'low resolution first and then full resolution.'
                                                               'We did not use it.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for processing the data')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='decay the learning rate after these many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--savedir', default='./results_ynet', help='results directory')
    parser.add_argument('--visualizeNet', type=bool, default=True, help='Visualize the network as PDF file')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Use this flag to load the last checkpoint for training')
    parser.add_argument('--resumeLoc', default='./results_ynet_C1/checkpoint.pth.tar', help='checkpoint location')
    parser.add_argument('--classes', type=int, default=8, help='Number of classes in the dataset')
    parser.add_argument('--diagClasses', type=int, default=5, help='Number of diagnostic classes')
    parser.add_argument('--cached_data_file', default='ynet_joint_cache.p', help='Data file names and other values, such as'
                                                                           'class weights, are cached')
    parser.add_argument('--logFile', default='trainValLog.txt', help="Log file")
    parser.add_argument('--onGPU', default=True, help='True if you want to train on GPU')
    parser.add_argument('--modelType', default='C1', help='Model could be C1 or D1')

    args = parser.parse_args()
    assert args.modelType in ['C1', 'D1']
    args.savedir = args.savedir + '_' + args.modelType + os.sep  # update the save dir name with model type

    # pre-trained segmentation model
    if args.modelType == 'C1':
        args.pretrainedSeg = '../stage1/pretrained_models_st1/model_C1.pth'
    else:
        args.pretrainedSeg = '../stage1/pretrained_models_st1/model_D1.pth'

    if not os.path.isfile(args.pretrainedSeg):
        args.pretrainedSeg = None

    print(args)
    trainValidateSegmentation(args)
