import os, cv2
import numpy as np
from network_configure import conf_unet
from network import *
from utils.predict_utils import get_coord, PercentileNormalizer, PadAndCropResizer
from utils.train_utils import augment_patch
from utils import train_utils

# UNet implementation inherited from GVTNets: https://github.com/zhengyang-wang/GVTNets
training_config = {'base_learning_rate': 0.0004,
                                     'lr_decay_steps':5e3, 
                                     'lr_decay_rate':0.5, 
                                     'lr_staircase':True}

class Noise2Same(object):

    def __init__(self, base_dir, name, 
                 dim=2, in_channels=1, lmbd=None, 
                 masking='gaussian', mask_perc=0.5,
                 opt_config=training_config, **kwargs):

        self.base_dir = base_dir # model direction
        self.name = name # model name
        self.dim = dim # image dimension
        self.in_channels = in_channels # image channels
        self.lmbd = lmbd # lambda in loss fn
        self.masking = masking
        self.mask_perc = mask_perc
        
        self.opt_config = opt_config
        conf_unet['dimension'] = '%dD'%dim
        self.net = UNet(conf_unet)
        
    def _model_fn(self, features, labels, mode):
        conv_op = convolution_2D if self.dim==2 else convolution_3D
        axis = {3:[1,2,3,4], 2:[1,2,3]}[self.dim]
        
        def image_summary(img):
            return tf.reduce_max(img, axis=1) if self.dim == 3 else img
        
        # Local average excluding the center pixel (donut)
        def mask_kernel(features):
            kernel = (np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]]) 
                      if self.dim == 2 else 
                      np.array([[[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                                [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]]]))
            kernel = (kernel/kernel.sum())
            kernels = np.empty([3, 3, self.in_channels, self.in_channels])
            for i in range(self.in_channels):
                kernels[:,:,i,i] = kernel
            nn_conv_op = tf.nn.conv2d if self.dim == 2 else tf.nn.conv3d
            return nn_conv_op(features, tf.constant(kernels.astype('float32')), 
                              [1]*self.dim+[1,1], padding='SAME')
        
        if not mode == tf.estimator.ModeKeys.PREDICT:
            noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)
            
            if self.masking == 'gaussian':
                masked_features = (1 - mask) * features + mask * noise
            elif self.masking == 'donut':
                masked_features = (1 - mask) * features + mask * mask_kernel(features)
            else:
                raise NotImplementedError
            
            # Prediction from masked input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                out = self.net(masked_features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name = 'out_conv')
                
            # Prediction from full input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                rawout = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                rawout = batch_norm(rawout, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                rawout = relu(rawout)
                rawpreds = conv_op(rawout, self.in_channels, 1, 1, False, name = 'out_conv')
            
            # Loss components
            loss_mse = tf.reduce_mean(tf.square(rawpreds-features), axis=None)
            inv_mse = tf.reduce_sum(tf.square(rawpreds - preds)*mask) / tf.reduce_sum(mask)
            rec_mse = tf.reduce_sum(tf.square(features - preds)*mask) / tf.reduce_sum(mask)

            # Tensorboard display
            tf.summary.image('1_inputs', image_summary(features), max_outputs=3)
            tf.summary.image('2_raw_predictions', image_summary(rawpreds), max_outputs=3)
            tf.summary.image('3_mask', image_summary(mask), max_outputs=3)
            tf.summary.image('4_masked_predictions', image_summary(preds), max_outputs=3)
            tf.summary.image('5_difference', image_summary(rawpreds-preds), max_outputs=3)
            tf.summary.image('6_rec_error', image_summary(preds-features), max_outputs=3)
            tf.summary.scalar('reconstruction', rec_mse, family='loss_metric') 
            tf.summary.scalar('invariance', inv_mse, family='loss_metric') 
            tf.summary.scalar('global', loss_mse, family='loss_metric')
                
        else:
            with tf.variable_scope('main_unet'):
                out = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name = 'out_conv')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=preds)
        
        lmbd = 2 if self.lmbd is None else self.lmbd
        loss = loss_mse + lmbd*tf.sqrt(inv_mse)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.opt_config['base_learning_rate'], 
                                                       global_step, 
                                                       self.opt_config['lr_decay_steps'], 
                                                       self.opt_config['lr_decay_rate'], 
                                                       self.opt_config['lr_staircase'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main_unet')
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        else:
            train_op = None
        
        metrics = {'loss_metric/invariance':tf.metrics.mean(inv_mse),
                              'loss_metric/reconstruction':tf.metrics.mean(rec_mse), 
                              'loss_metric/global':tf.metrics.mean(loss_mse)}
        
        return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op, 
                                          eval_metric_ops=metrics)


    def _input_fn(self, sources, patch_size, batch_size, is_train=True):
        # Stratified sampling inherited from Noise2Void: https://github.com/juglab/n2v
        get_stratified_coords = getattr(train_utils, 'get_stratified_coords%dD'%self.dim)
        rand_float_coords = getattr(train_utils, 'rand_float_coords%dD'%self.dim)
        
        def generator():
            while(True):
                source = sources[np.random.randint(len(sources))]
                valid_shape = source.shape[:-1] - np.array(patch_size)
                if any([s<=0 for s in valid_shape]):
                    source_patch = augment_patch(source)
                else:
                    coords = [np.random.randint(0, shape_i+1) for shape_i in valid_shape]
                    s = tuple([slice(coord, coord+size) for coord, size in zip(coords, patch_size)])
                    source_patch = augment_patch(source[s])
                
                mask = np.zeros_like(source_patch)
                for c in range(self.in_channels):
                    boxsize = np.round(np.sqrt(100/self.mask_perc)).astype(np.int)
                    maskcoords = get_stratified_coords(rand_float_coords(boxsize), 
                                                       box_size=boxsize, shape=tuple(patch_size))
                    indexing = maskcoords + (c,)
                    mask[indexing] = 1.0

                noise_patch = np.concatenate([np.random.normal(0, 0.2, source_patch.shape), mask], axis=-1)
                yield source_patch, noise_patch
                
        def generator_val():
            for idx in range(len(sources)):
                source_patch = sources[idx]
                patch_size = source_patch.shape[:-1]
                boxsize = np.round(np.sqrt(100/self.mask_perc)).astype(np.int)
                maskcoords = get_stratified_coords(rand_float_coords(boxsize), 
                                                   box_size=boxsize, shape=tuple(patch_size))
                indexing = maskcoords + (0,)
                mask = np.zeros_like(source_patch)
                mask[indexing] = 1.0
                noise_patch = np.concatenate([np.random.normal(0, 0.2, source_patch.shape), mask], axis=-1)
                yield source_patch, noise_patch

        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape(list(patch_size) + [self.in_channels]), 
                                             tf.TensorShape(list(patch_size) + [self.in_channels*2]))
        gen = generator if is_train else generator_val
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    def train(self, source_lst, patch_size, validation=None, batch_size=64, save_steps=1000, log_steps=200, steps=50000):
        assert len(patch_size)==self.dim
        assert len(source_lst[0].shape)==self.dim + 1
        assert source_lst[0].shape[-1]==self.in_channels

        ses_config = tf.ConfigProto()
        ses_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(model_dir=self.base_dir+'/'+self.name, 
                                            save_checkpoints_steps=save_steps,
                                            session_config=ses_config, 
                                            log_step_count_steps=log_steps,
                                            save_summary_steps=log_steps,
                                            keep_checkpoint_max=2)

        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                             model_dir=self.base_dir+'/'+self.name, 
                                             config=run_config)
        
        input_fn = lambda: self._input_fn(source_lst, patch_size, batch_size=batch_size)
        
        if validation is not None:
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=steps)
            val_input_fn = lambda: self._input_fn(validation.astype('float32'), 
                                                  validation.shape[1:-1], 
                                                  batch_size=4, 
                                                  is_train=False)
            eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, throttle_secs=120)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            estimator.train(input_fn=input_fn, steps=steps)
            

    # Used for single image prediction
    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        image = (image - im_mean)/im_std
        if self.in_channels == 1:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            image = resizer.after(image, exclude=None)
        else:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=-1)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None], batch_size=1, num_epochs=1, shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            image = resizer.after(image, exclude=-1)
        image = image*im_std + im_mean

        return image
    
    # Used for batch images prediction
    def batch_predict(self, images, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None, batch_size=32):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((images.mean(), images.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        
        images = (images - im_mean)/im_std
        images = resizer.before(images, 2 ** (self.net.depth), exclude=0)
        input_fn = tf.estimator.inputs.numpy_input_fn(x=images[ ..., None], batch_size=batch_size, num_epochs=1, shuffle=False)
        images = np.stack(list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)))[..., 0]
        images = resizer.after(images, exclude=0)
        images = images*im_std + im_mean

        return images

    # Used for extremely large input images
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        image = (image - im_mean)/im_std
        out_image = np.empty(image.shape, dtype='float32')
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            patch = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]
            
        image = out_image*im_std + im_mean

        return image