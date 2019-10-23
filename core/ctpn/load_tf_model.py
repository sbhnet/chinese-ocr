import tensorflow as tf
from .lib.fast_rcnn.config import cfg
from .lib.networks.factory import get_network


def load_tf_model(path):
    # load config file
    cfg.TEST.checkpoints_path = path

    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)

    # load network
    net = get_network("VGGnet_test")

    # load model
    print('Loading network {:s}... '.format("VGGnet_test"))
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    return sess, net
