import caffe
from caffe import layers as L
from caffe import params as P
import common

def batch_norm(bottom_attr, n ,prefix, prefix_1, appendix, phase):
    bottom = getattr(n, bottom_attr)
    if phase == "Train":
        setattr(n, prefix_1 + "bn" + prefix + appendix, L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=False),
                                                   param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True))
        setattr(n, prefix_1 + "scale" +prefix + appendix, L.Scale(bottom, scale_param=dict(bias_term=True), in_place=True))
    if phase == "Test":
        setattr(n, prefix_1 + "bn" + prefix + appendix, L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=True),
                                                   param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True))
        setattr(n, prefix_1 + "scale" + prefix + appendix, L.Scale(bottom, scale_param=dict(bias_term=True), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], in_place=True))

    return

def resnet_block_1_2_fused(bottom_attr, n, prefix, prefix_1, phase, num_o_1, num_o_2, num_o_3, learn = True, dilation = 1, dropout_ratio = 0, stride_1 = 1, dilation_pool = 1):
    bottom = getattr(n, bottom_attr)
    conv1_3_attr = resnet_block_1(bottom_attr, n, prefix, prefix_1, phase, num_o_1, num_o_2, num_o_3, learn, dilation, dropout_ratio, stride_1, dilation_pool)
    conv_1_3 = getattr(n, conv1_3_attr)

    setattr(n, prefix_1 + "res" + prefix,L.Eltwise(conv_1_3, bottom))

    elt_1_p = getattr(n, prefix_1+ "res" + prefix)
    setattr(n, prefix_1 + "res" + prefix + "_relu", L.ReLU(elt_1_p, in_place=True))

    return prefix_1+ "res" + prefix

def resent_block_1_1_fused(bottom_attr,n,prefix, prefix_1, phase, num_o_1, num_o_2, num_o_3, learn = True, dilation = 1, dropout_ratio = 0.0, stride_1 = 1, dilation_pool = 1):
    bottom = getattr(n, bottom_attr)
    conv1_3_attr = resnet_block_1(bottom_attr, n, prefix, prefix_1, phase, num_o_1, num_o_2, num_o_3, learn, dilation, dropout_ratio, stride_1, dilation_pool)
    conv1_3 = getattr(n, conv1_3_attr)

    # Parallel Convolution
    setattr(n, prefix_1+  "res" + prefix + "_branch1", L.Convolution(bottom, kernel_size=1, num_output=num_o_3, pad=0, stride=stride_1,
                                                 param=dict(lr_mult=common.lw, decay_mult=common.lw), bias_term=False,
                                                 engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
    batch_norm(prefix_1+ "res" +prefix+"_branch1", n,prefix, prefix_1, "_branch1",  phase)

    # Fuse Parts
    conv1_p = getattr(n, prefix_1 + "res" + prefix + "_branch1")
    setattr(n, prefix_1 + "res" + prefix, L.Eltwise(conv1_3, conv1_p))
    elt_1_p = getattr(n, prefix_1 + "res" + prefix)
    setattr(n, prefix_1 + "res" + prefix + "_relu", L.ReLU(elt_1_p, in_place = True))
    return prefix_1 + "res" + prefix

def resnet_block_1(bottom_attr, n, prefix, prefix_1, phase, num_o_1, num_o_2, num_o_3, learn = True, dilation = 1, dropout_ratio=0.0, stride_1 = 1, dilation_pool = 1):
    lw = 0
    lb = 0

    if(learn):
        lw = 1
        lb = 2

    bottom = getattr(n, bottom_attr)
    # Convolution 1
    setattr(n, prefix_1 + "res" + prefix + "_branch2a", L.Convolution(bottom, kernel_size=1, num_output=num_o_1, pad=0, stride = stride_1,
                                                 param=dict(lr_mult=lw, decay_mult=lw), bias_term = False,
                                                 engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
    batch_norm( prefix_1 + "res" + prefix + "_branch2a",n,prefix, prefix_1, "_branch2a", phase)

    conv1_1 = getattr(n, prefix_1 + "res" + prefix + "_branch2a")
    setattr(n, prefix_1 + "res" + prefix + "_branch2a_relu", L.ReLU(conv1_1, in_place=True))

    # Convolution 2
    # Is this a dilated pooling?
    if dilation_pool > 1:
        if dilation > 1:
            setattr(n, prefix_1 + "res" + prefix + "conv1_2_p1", L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2/2, pad=dilation_pool, stride=1, dilation = dilation_pool,
                                                            param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                                            engine=P.Convolution.CAFFE, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res"+ prefix+"conv1_2_p1", n, prefix, prefix_1, "_dil1", phase)

            conv1_2_p1 = getattr(n, prefix_1 + "res" + prefix + "conv1_2_p1")
            setattr(n, prefix_1 + "res" + prefix + "relu1_2", L.ReLU(conv1_2_p1, in_place=True))

            setattr(n, prefix_1 + "res" + prefix + "conv1_2_p2",L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2/2, pad=dilation, dilation=dilation,
                                                           param=[dict(lr_mult=lw, decay_mult=lw), dict(lr_mult=lb, decay_mult=0)],
                                                           engine=P.Convolution.CAFFE, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res" + prefix + "conv1_2_p2",n,prefix, prefix_1,"_dil2", phase)

            conv1_2_p2 = getattr(n, prefix_1 + "res"+ prefix + "conv1_2_p2")
            setattr(n, prefix_1 + "res" + prefix + "relu1_2_p2", L.ReLU(conv1_2_p2, in_place=True))
            #setattr(n, prefix + "dil_sum", L.Eltwise(conv1_2_p1, conv1_2_p2, operation=P.Eltwise.SUM))
            setattr(n, prefix_1 +  prefix + "dil_sum", L.Concat(conv1_2_p1, conv1_2_p2))
        else:
            setattr(n,prefix_1 + "res" + prefix + "_branch2b", L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2, pad=dilation_pool, stride=1, dilation = dilation_pool,
                                                            param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                                            engine=P.Convolution.CAFFE, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res" + prefix + "_branch2b", n, prefix, prefix_1, "_branch2b", phase)
            conv1_2_p1 = getattr(n, prefix_1 + "res" + prefix + "_branch2b")
            setattr(n, prefix_1 + "res"+ prefix + "_branch2b_relu", L.ReLU(conv1_2_p1, in_place=True))
    else:
        if dilation > 1:
            setattr(n, prefix_1 + "res" + prefix + "conv1_2_p1",
                    L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2 / 2, pad=1, stride=1,
                                  param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                  engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res" + prefix + "conv1_2_p1", n, prefix, prefix_1, "_dil1", phase)

            conv1_2_p1 = getattr(n, prefix_1 + "res" + prefix + "conv1_2_p1")
            setattr(n, prefix_1 + "res" + prefix + "relu1_2", L.ReLU(conv1_2_p1, in_place=True))

            setattr(n, prefix_1 + "res" + prefix + "conv1_2_p2",
                    L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2 / 2, pad=dilation, dilation=dilation,
                                  param=[dict(lr_mult=lw, decay_mult=lw), dict(lr_mult=lb, decay_mult=0)],
                                  engine=P.Convolution.CAFFE, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res" + prefix + "conv1_2_p2", n, prefix, prefix_1, "_dil2", phase)

            conv1_2_p2 = getattr(n, prefix_1+ "res" + prefix + "conv1_2_p2")
            setattr(n, prefix_1 + "res" + prefix + "relu1_2_p2", L.ReLU(conv1_2_p2, in_place=True))
            #setattr(n, prefix + "dil_sum", L.Eltwise(conv1_2_p1, conv1_2_p2, operation=P.Eltwise.SUM))
            setattr(n, prefix_1 + prefix + "dil_sum", L.Concat(conv1_2_p1, conv1_2_p2))
        else:
            setattr(n, prefix_1 + "res" + prefix + "_branch2b",
                    L.Convolution(conv1_1, kernel_size=3, num_output=num_o_2, pad=1, stride=1,
                                  param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                  engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
            batch_norm(prefix_1 + "res" + prefix + "_branch2b", n, prefix, prefix_1, "_branch2b", phase)
            conv1_2_p1 = getattr(n, prefix_1 + "res" + prefix + "_branch2b")
            setattr(n, prefix_1 + "res" + prefix + "_branch2b_relu", L.ReLU(conv1_2_p1, in_place=True))


    # Convolution 3
    if(dilation > 1):
        dil_sum = getattr(n, prefix_1 + prefix + "dil_sum")
        setattr(n, prefix_1 + "res" + prefix + "_branch2c", L.Convolution(dil_sum, kernel_size=1, num_output=num_o_3, pad=0, stride=1,
                                                     param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                                     engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
    else:
        setattr(n, prefix_1 + "res" + prefix + "_branch2c", L.Convolution(conv1_2_p1, kernel_size=1, num_output=num_o_3, pad=0, stride=1,
                                                     param=dict(lr_mult=lw, decay_mult=lw), bias_term=False,
                                                     engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))

    if dropout_ratio > 0.0:
        conv1_3 = getattr(n, prefix_1 + "res" + prefix + "_branch2c")
        setattr(n, prefix_1 + prefix + "dropout_2", L.Dropout(conv1_3, dropout_param=dict(dropout_ratio=dropout_ratio), in_place=True))

    batch_norm(prefix_1 + "res" + prefix+"_branch2c",n, prefix, prefix_1, "_branch2c", phase)

    return prefix_1 + "res" + prefix+"_branch2c"


def conv_expan(bottom_attr,n,prefix, prefix_1, nout, ks=1, stride=1, pad=0, init_w = False):
    bottom = getattr(n, bottom_attr)
    if init_w:
        setattr(n, prefix_1 + prefix + "conv_1", L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                                                  param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                                                  engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
        return prefix_1 + prefix + "conv_1"
    else:
        setattr(n, prefix_1 + prefix + "conv_1", L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                                                  param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                                                  engine=P.Convolution.CUDNN))
        return prefix_1 + prefix + "conv_1"

def deconv_expan(bottom_attr,n,prefix, prefix_1, nout, ks=4, stride=2, pad=0, learn = True):
    bottom = getattr(n, bottom_attr)
    if learn:
        setattr(n, prefix_1 + prefix + "deconv_1", L.Deconvolution(bottom, param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                                                      convolution_param=dict(kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                                                                             weight_filler=dict(type='bilinear'))))
        return prefix_1 + prefix + "deconv_1"
    else:
        setattr(n, prefix_1 + prefix + "deconv_1", L.Deconvolution(bottom, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                                                      convolution_param=dict(kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                                                                             weight_filler=dict(type='bilinear'))))
        return prefix_1 + prefix + "deconv_1"

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def multinet(data, n, phase, pref):

    learn_blocks = False
    if(common.lw > 0):
        learn_blocks = True

    setattr(n, pref + "conv1_0", L.Convolution(data, kernel_size=3, num_output=32, pad=1, stride=1,
                              param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                              engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
    conv1_0 = getattr(n, pref + "conv1_0")
    batch_norm(pref + "conv1_0", n, "init1_0__", pref, "egal", phase)
    setattr(n, pref + "relu1_0", L.ReLU(conv1_0, in_place=True))

    # Conv, Batch Norm, Max Pool
    setattr(n, pref + "conv1_2",L.Convolution(conv1_0, kernel_size=7, num_output=64, pad=3, stride=2,
                              param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                              engine=P.Convolution.CUDNN, weight_filler=dict(type='xavier')))
    conv1_2 = getattr(n, pref + "conv1_2")
    batch_norm(pref + "conv1_2", n, "", pref, "_conv1_2", phase)
    setattr(n, pref + "conv1_2_relu",L.ReLU(conv1_2, in_place=True))
    setattr(n, pref + "pool1_2", L.Pooling(conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0))

    # COMPRESSION SIDE
    # Resnet Blocks: 1
    elt_1 = resent_block_1_1_fused(pref + "pool1_2", n, "2a", pref, phase, 64 / common.feat_div, 64 / common.feat_div , 256 / common.feat_div, learn_blocks)
    elt_2 = resnet_block_1_2_fused(elt_1, n, "2b", pref, phase, 64 / common.feat_div, 64 / common.feat_div, 256 / common.feat_div, learn_blocks)
    elt_3 = resnet_block_1_2_fused(elt_2, n, "2c", pref, phase, 64 / common.feat_div, 64 / common.feat_div, 256 / common.feat_div, learn_blocks)

    # Resnet Blocks: 2
    elt_4 = resent_block_1_1_fused(elt_3, n, "3a", pref, phase, 128 / common.feat_div, 128 / common.feat_div, 512 / common.feat_div,learn_blocks, 1, 0, 2)
    elt_5 = resnet_block_1_2_fused(elt_4, n, "3b", pref, phase, 128 / common.feat_div, 128 / common.feat_div, 512 / common.feat_div, learn_blocks)
    elt_6 = resnet_block_1_2_fused(elt_5, n, "3c", pref, phase, 128 / common.feat_div, 128 / common.feat_div, 512 / common.feat_div,learn_blocks, 1, 0, 1)
    elt_7 = resnet_block_1_2_fused(elt_6, n, "3d", pref, phase, 128 / common.feat_div, 128 / common.feat_div, 512 / common.feat_div,learn_blocks, 2, 0, 1)

    # Resnet Blocks: 3
    elt_8 = resent_block_1_1_fused(elt_7, n, "4a", pref, phase, 256 / common.feat_div, 256 / common.feat_div, 1024 / common.feat_div, learn_blocks, 1, 0, 2)
    elt_9 = resnet_block_1_2_fused(elt_8, n, "4b", pref, phase, 256 / common.feat_div, 256 / common.feat_div, 1024 / common.feat_div, learn_blocks, 1, 0, 1)
    elt_10 = resnet_block_1_2_fused(elt_9, n, "4c", pref, phase, 256 / common.feat_div, 256 / common.feat_div, 1024 / common.feat_div, learn_blocks, 2, 0, 1)

    # Resnet Blocks: 4
    elt_14 = resent_block_1_1_fused(elt_10, n, "5a", pref, phase, 512 / common.feat_div, 512 / common.feat_div, 2048 / common.feat_div, learn_blocks, 4, 0, 1, 2)
    elt_15 = resnet_block_1_2_fused(elt_14, n, "5b", pref, phase, 512 / common.feat_div, 512 / common.feat_div, 2048 / common.feat_div, learn_blocks, 8, 0, 1, 2)
    elt_16 = resnet_block_1_2_fused(elt_15, n, "5c", pref, phase, 512 / common.feat_div, 512 / common.feat_div, 2048 / common.feat_div, learn_blocks, 16, 0.45, 1, 2)

    # EXPANSION SIDE
    score_fr = conv_expan(elt_16, n, "expan1", pref, common.c, 1, 1, 0, True)
    batch_norm(score_fr,n,"dec1", pref, "_1", phase)

    score2 = deconv_expan(score_fr,n,"deconv1_", pref, common.c * common.N, 4, 2, 1)
    batch_norm(score2, n, "dec2", pref, "_1", phase)

    elt2_fine =  conv_expan(elt_7,n,"fine_2_", pref, common.c * common.N, 1, 1, 0)
    batch_norm(elt2_fine, n, "dec5", pref , "_1", phase)

    score4_attr = getattr(n, score2)
    elt2_fine_attr = getattr(n, elt2_fine)
    setattr(n, pref + "score_c2",L.Eltwise(score4_attr, elt2_fine_attr, operation=P.Eltwise.SUM))
    score_c2 = getattr(n, pref + "score_c2")

    return score_c2

def buildExecutableNet(lmdb_images, lmdb_labels, batch_size, phase):
    n = caffe.NetSpec()

    pref = common.layer_prefix

    # LOAD DATA
    n.data = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_images,
                    transform_param=dict(mean_file=common.mean_file))
    n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_labels)

    if phase == "Test":
        n.data = L.Data(batch_size=1, backend=P.Data.LMDB, source=lmdb_images,
                        transform_param=dict(mean_file=common.mean_file))
        n.label = L.Data(batch_size=1, backend=P.Data.LMDB, source=lmdb_labels)

    # Embed net....
    output = multinet(n.data, n, phase, pref)

    # Upsample to full size (fixed weights upsampling)
    setattr(n, pref + "score_temp3",
            L.Deconvolution(output, param=[dict(lr_mult=common.lw, decay_mult=common.lw), dict(lr_mult=common.lb, decay_mult=0)],
                            convolution_param=dict(kernel_size=16, stride=8, num_output=common.c, pad=4,
                                                   weight_filler=dict(type='bilinear'))))
    score_temp3 = getattr(n, pref + "score_temp3")
    batch_norm(pref + "score_temp3", n, "temp3", pref, "", phase)


    if phase == "Train":
        n.loss = L.SoftmaxWithLoss(score_temp3, n.label, loss_param= dict(normalize = False, ignore_label = common.ig_lbl))
    if phase == "Test":
        n.score_argmax = L.ArgMax(score_temp3, argmax_param= dict(axis = 1))
        n.class_iou = L.IntersectionOverUnion(n.score_argmax, n.label, parse_iou_param= dict(num_labels = common.c, ignore_label = common.ig_lbl, total_im_num = common.testset_size))

    return n.to_proto()
