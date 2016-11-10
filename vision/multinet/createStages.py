import common
import multinet
import solver_spec

with open(common.output_dir + common.file_prefix + '3.prototxt', 'w') as f:
    f.write(str(multinet.buildExecutableNet(common.images_db_path, common.labels_db_path, common.batch_size, "Train")))

with open(common.output_dir + common.file_prefix + '3_test.prototxt', 'w') as f:
    f.write(str(multinet.buildExecutableNet(common.test_images_db_path, common.test_labels_db_path, common.batch_size, "Test")))

with open(common.output_dir + common.file_prefix + 'solver_3.prototxt', 'w') as f:
    f.write(solver_spec.define_solver(common.file_prefix + '3.prototxt', common.file_prefix + '3_test.prototxt', 'snapshots/train', common.test_at_iter, "1e-6"))

