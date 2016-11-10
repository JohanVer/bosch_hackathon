import common

def define_solver(train_net, test_net, snapshot_dir, test_at_iter, base_lr):

    solver = ""
    solver = solver + "train_net: \"" + train_net + "\""
    solver = solver + "\ntest_net: \"" + test_net + "\""
    solver = solver + "\ntest_iter: " + str(common.testset_size)
    solver = solver + "\ntest_interval: " + str(test_at_iter)
    solver = solver + "\ndisplay: 20"
    solver = solver + "\naverage_loss: 20"
    solver = solver + "\nlr_policy: \"poly\" "
    solver = solver + "\npower: 0.9"
    solver = solver + "\nbase_lr: " + base_lr
    solver = solver + "\nmomentum: 0.9"
    solver = solver + "\niter_size: 1"
    solver = solver + "\nmax_iter: 40000"
    solver = solver + "\nweight_decay: 0.0005"
    solver = solver + "\nsnapshot: " + str(test_at_iter)
    solver = solver + "\nsnapshot_prefix: \"" + snapshot_dir + "\""
    solver = solver + "\ntest_initialization: false"

    return solver