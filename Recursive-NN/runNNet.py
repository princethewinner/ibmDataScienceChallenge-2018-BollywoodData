import optparse
import cPickle as pickle

import sgd as optimizer
from rnn2deep import RNN2
import tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb

# Helper to generate a confusion matrix
from sklearn.metrics import confusion_matrix


# This is the main training function of the codebase. You are intended to run this function via command line
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)


    parser.add_option("--middleDim",dest="middleDim",type="int",default=10)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=3)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    # for DCNN only
    parser.add_option("--ktop",dest="ktop",type="int",default=5)
    parser.add_option("--m1",dest="m1",type="int",default=10)
    parser.add_option("--m2",dest="m2",type="int",default=7)
    parser.add_option("--n1",dest="n1",type="int",default=6)
    parser.add_option("--n2",dest="n2",type="int",default=12)

    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    (opts,args)=parser.parse_args(args)

    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        cmfile = opts.inFile + ".confusion_matrix-" + opts.data
        test(opts.inFile,opts.data,None,opts.model,confusion_matrix_file=cmfile,full=True)
        return

    print "Loading data..."
    train_accuracies = []
    dev_accuracies = []
    # load training data
    trees = tr.loadTrees('train')
    opts.numWords = len(tr.loadWordMap())

    #Load word embeddings
    L = tr.loadWordEmbedding()

    if(opts.model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    else:
        raise '%s is not a valid neural network, only RNN2'%opts.model

    nn.initParams(L)

    sgd = optimizer.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)


    dev_trees = tr.loadTrees("dev")
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d"%e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)
            nn.toFile(fid)
        if evaluate_accuracy_while_training:
            print "testing on training set real quick"
            train_accuracies.append(test(opts.outFile,"train",L,opts.model,trees))
            print "testing on dev set real quick"
            dev_accuracies.append(test(opts.outFile,"dev",L,opts.model,dev_trees))
            # clear the fprop flags in trees and dev_trees
            for tree in trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            for tree in dev_trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            print "fprop in trees cleared"


    if evaluate_accuracy_while_training:
        # pdb.set_trace()
        print train_accuracies
        print dev_accuracies
        # Plot train/dev_accuracies here?
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, label='Train')
        plt.plot(range(len(dev_accuracies)), dev_accuracies, label='Dev')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # plot.show()
        plt.savefig(opts.outFile + ".accuracy_plot.png")



def test(netFile, dataSet, L, model='RNN', trees=None, confusion_matrix_file=None, full=False):
    if trees==None:
        trees = tr.loadTrees(dataSet)
    if L is None:
        L = tr.loadWordEmbedding()
    assert netFile is not None, "Must give model to test"
    print "Testing netFile %s"%netFile
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)

        if(model=='RNN2'):
            nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        else:
            raise '%s is not a valid neural network , only RNN2'%opts.model

        nn.initParams(L)
        nn.fromFile(fid)

    print "Testing %s..."%model

    cost, correct, guess, total, actss = nn.costAndGrad(trees,test=True)
    if full:
        #pass
        import pickle as pkl
        with open('{}_actss_{}.pkl'.format(netFile, dataSet),'w') as fid:
            pkl.dump(actss,fid)

    correct_sum = 0
    for i in xrange(0,len(correct)):
        correct_sum+=(guess[i]==correct[i])

    # Generate confusion matrix
    if confusion_matrix_file is not None:
        cm = confusion_matrix(correct, guess)
        makeconf(cm, confusion_matrix_file)

    print "Cost %f, Acc %f"%(cost,correct_sum/float(total))
    return correct_sum/float(total)


def makeconf(conf_arr, outfile):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(5, 5))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res, shrink=0.5)
    #indexs = '0123456789'
    indexs = ['Male', 'Female', 'Neutral']
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])

    # you can save the figure here with:
    #plt.savefig(outfile)
    plt.savefig(outfile+'.svg', format='svg')
    print "Confusion Matrix written to %s" % outfile
    plt.show()


if __name__=='__main__':
    run()


