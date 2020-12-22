import tensorflow as tf
import numpy as np
import torch

class ModelAdapter():
    def __init__(self, model, num_classes=10):
        """
        Please note that model should be tf.keras model without activation function 'softmax'
        """
        self.num_classes = num_classes
        self.tf_model = model
        self.__check_channel_ordering()

    def __check_channel_ordering(self):
        
        for L in self.tf_model.layers:
            if isinstance(L, tf.keras.layers.Conv2D):
                print("[INFO] set data_format = '{:s}'".format(L.data_format))
                self.data_format = L.data_format
                return

        print("[INFO] Can not find Conv2D layer")
        input_shape = self.tf_model.input_shape

        if input_shape[3] == 3:
            print("[INFO] Because detecting input_shape[3] == 3, set data_format = 'channels_last'")
            self.data_format = 'channels_last'

        elif input_shape[3] == 1:
            print("[INFO] Because detecting input_shape[3] == 1, set data_format = 'channels_last'")
            self.data_format = 'channels_last'

        else:
            print("[INFO] set data_format = 'channels_first'")
            self.data_format = 'channels_first'

    def __get_logits(self, x_input):
        logits = self.tf_model(x_input, training=False)
        return logits

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_jacobian(self, x_input):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)

        jacobian = g.batch_jacobian(logits, x_input)

        if self.data_format == 'channels_last':
            jacobian = tf.transpose(jacobian, perm=[0,1,4,2,3])

        return jacobian

    def __get_xent(self, logits, y_input):
        xent   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
        return xent

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_xent(self, x_input, y_input):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            xent = self.__get_xent(logits, y_input)
        
        grad_xent = g.gradient(xent, x_input)

        return logits, xent, grad_xent

    def __get_dlr(self, logits, y_input):
        val_dlr = dlr_loss(logits, y_input, num_classes=self.num_classes)
        return val_dlr

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_dlr(self, x_input, y_input):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            val_dlr = self.__get_dlr(logits, y_input)

        grad_dlr = g.gradient(val_dlr, x_input)
        
        return logits, val_dlr, grad_dlr

    def __get_dlr_target(self, logits, y_input, y_target):
        dlr_target = dlr_loss_targeted(logits, y_input, y_target, num_classes=self.num_classes)
        return dlr_target

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_dlr_target(self, x_input, y_input, y_target):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            dlr_target = self.__get_dlr_target(logits, y_input, y_target)

        grad_target = g.gradient(dlr_target, x_input)

        return logits, dlr_target, grad_target
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_diff_logits_target(self, x, la, la_target):
        la_mask = tf.one_hot(la, self.num_classes)
        la_target_mask = tf.one_hot(la_target, self.num_classes)
        
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x)
            logits = self.__get_logits(x)
            difflogits = tf.reduce_sum((la_target_mask - la_mask) * logits, axis=1)
    
        g2 = g.gradient(difflogits, x)
        
        return difflogits, g2
    
    def predict(self, x):

        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        y = self.__get_logits(x2).numpy()
        
        return torch.from_numpy(y).cuda()

    def grad_logits(self, x):
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])
        g2 = self.__get_jacobian(x2)
        
        return torch.from_numpy(g2.numpy()).cuda()

    def set_target_class(self, y, y_target):
        pass
    
    def get_grad_diff_logits_target(self, x, y, y_target):
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])
        la = y.cpu().numpy()
        la_target = y_target.cpu().numpy()
        
        difflogits, g2 = self.__get_grad_diff_logits_target(x2, la, la_target)
            
        if self.data_format == 'channels_last':
            g2 = tf.transpose(g2, perm=[0, 3, 1, 2])
        
        return torch.from_numpy(difflogits.numpy()).cuda(), torch.from_numpy(g2.numpy()).cuda()
    
    def get_logits_loss_grad_xent(self, x, y):

        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.clone().cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        logits_val, loss_indiv_val, grad_val = self.__get_grad_xent(x2, y2)

        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        
        return torch.from_numpy(logits_val.numpy()).cuda(), torch.from_numpy(loss_indiv_val.numpy()).cuda(), torch.from_numpy(grad_val.numpy()).cuda()

    def get_logits_loss_grad_dlr(self, x, y):

        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.clone().cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        logits_val, loss_indiv_val, grad_val = self.__get_grad_dlr(x2, y2)

        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        
        return torch.from_numpy(logits_val.numpy()).cuda(), torch.from_numpy(loss_indiv_val.numpy()).cuda(), torch.from_numpy(grad_val.numpy()).cuda()
    
    def get_logits_loss_grad_target(self, x, y, y_target):

        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.clone().cpu().numpy(), dtype=tf.int32)
        y_targ = tf.convert_to_tensor(y_target.clone().cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        logits_val, loss_indiv_val, grad_val = self.__get_grad_dlr_target(x2, y2, y_targ)

        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        
        return torch.from_numpy(logits_val.numpy()).cuda(), torch.from_numpy(loss_indiv_val.numpy()).cuda(), torch.from_numpy(grad_val.numpy()).cuda()

def dlr_loss(x, y, num_classes=10):

    # logit
    logit = x
    logit_sort = tf.sort(logit, axis=1)

    # onthot_y
    y_onehot = tf.one_hot(y , num_classes, dtype=tf.float32)
    logit_y = tf.reduce_sum(y_onehot * logit, axis=1)

    # z_i
    logit_pred = tf.reduce_max(logit, axis=1)
    cond = (logit_pred == logit_y)
    z_i = tf.where(cond, logit_sort[:, -2], logit_sort[:, -1])

    # loss
    z_y = logit_y
    z_p1 =  logit_sort[:, -1]
    z_p3 = logit_sort[:, -3]

    loss = - (z_y - z_i) / (z_p1 - z_p3 + 1e-12)

    return loss

def dlr_loss_targeted(x, y, y_target, num_classes=10):
    x_sort = tf.sort(x, axis=1)
    y_onehot = tf.one_hot(y, num_classes)
    y_target_onehot = tf.one_hot(y_target, num_classes)
    loss = -(tf.reduce_sum(x * y_onehot, axis=1) - tf.reduce_sum(x * y_target_onehot, axis=1)) / (x_sort[:, -1] - .5 * x_sort[:, -3] - .5 * x_sort[:, -4] + 1e-12)
    
    return loss
