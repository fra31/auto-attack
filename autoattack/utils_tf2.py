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
        self.data_format = self.__check_channel_ordering()

    def __tf_to_pt(self, tf_tensor):
        """ Private function
        Convert tf tensor to pt format

        Args:
            tf_tensor: (tf_tensor) TF tensor

        Retruns:
            pt_tensor: (pt_tensor) Pytorch tensor
        """

        cpu_tensor = tf_tensor.numpy()
        pt_tensor = torch.from_numpy(cpu_tensor).cuda()

        return pt_tensor

    def set_data_format(self, data_format):
        """
        Set data_format manually

        Args:
            data_format: A string, whose value should be either 'channels_last' or 'channels_first'
        """

        if data_format != 'channels_last' or data_format != 'channels_first':
            raise ValueError("data_format should be either 'channels_last' or 'channels_first'")

        self.data_format = data_format


    def __check_channel_ordering(self):
        """ Private function
        Determinate TF model's channel ordering based on model's information.
        Default ordering is 'channels_last' in TF.
        However, 'channels_first' is used in Pytorch.

        Returns:
            data_format: A string, whose value should be either 'channels_last' or 'channels_first'
        """

        data_format = None

        # Get the ordering of the dimensions in data from TF model
        for L in self.tf_model.layers:
            if isinstance(L, tf.keras.layers.Conv2D):
                print("[INFO] set data_format = '{:s}'".format(L.data_format))
                data_format = L.data_format
                break

        # Guess the ordering of the dimensions in data by input dimensions which sould be 4-D tensor
        if data_format is None:
            print("[WARNING] Can not find Conv2D layer")
            input_shape = self.tf_model.input_shape

            # Assume that input is *colorful image* whose dimensions should be [batch_size, img_w, img_h, 3]
            if input_shape[3] == 3:
                print("[INFO] Because detecting input_shape[3] == 3, set data_format = 'channels_last'")
                data_format = 'channels_last'

            # Assume that input is *gray image* whose dimensions should be [batch_size, img_w, img_h, 1]
            elif input_shape[3] == 1:
                print("[INFO] Because detecting input_shape[3] == 1, set data_format = 'channels_last'")
                data_format = 'channels_last'

            # Assume that input is *colorful image* whose dimensions should be [batch_size, 3, img_w, img_h]
            elif input_shape[1] == 3:
                print("[INFO] Because detecting input_shape[1] == 3, set data_format = 'channels_first'")
                data_format = 'channels_first'

            # Assume that input is *gray image* whose dimensions should be [batch_size, 1, img_w, img_h]
            elif input_shape[1] == 1:
                print("[INFO] Because detecting input_shape[1] == 1, set data_format = 'channels_first'")
                data_format = 'channels_first'

            else:
                print("[ERROR] Unknow case")

        return data_format


    # Common function which may be called in tf.function #
    def __get_logits(self, x_input):
        """ Private function
        Get model's pre-softmax output in inference mode

        Args:
            x_input: (tf_tensor) Input data

        Returns:
            logits: (tf_tensor) Logits
        """

        return self.tf_model(x_input, training=False)


    def __get_xent(self, logits, y_input):
        """ Private function
        Get cross entropy loss

        Args:
            logits: (tf_tensor) Logits.
            y_input: (tf_tensor) Label.

        Returns:
            xent: (tf_tensor) Cross entropy
        """

        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)


    def __get_dlr(self, logit, y_input):
        """ Private function
        Get DLR loss

        Args:
            logit: (tf_tensor) Logits
            y_input: (tf_tensor) Input label

        Returns:
            loss: (tf_tensor) DLR loss
        """

        # logit
        logit_sort = tf.sort(logit, axis=1)

        # onthot_y
        y_onehot = tf.one_hot(y_input , self.num_classes, dtype=tf.float32)
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


    def __get_dlr_target(self, logits, y_input, y_target):
        """ Private function
        Get targeted version of DLR loss

        Args:
            logit: (tf_tensor) Logits
            y_input: (tf_tensor) Input label
            y_target: (tf_tensor) Input targeted label

        Returns:
            loss: (tf_tensor) Targeted DLR loss
        """

        x = logits
        x_sort = tf.sort(x, axis=1)
        y_onehot = tf.one_hot(y_input, self.num_classes)
        y_target_onehot = tf.one_hot(y_target, self.num_classes)
        loss = -(tf.reduce_sum(x * y_onehot, axis=1) - tf.reduce_sum(x * y_target_onehot, axis=1)) / (x_sort[:, -1] - .5 * x_sort[:, -3] - .5 * x_sort[:, -4] + 1e-12)

        return loss


    # function called by public API directly #
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_jacobian(self, x_input):
        """ Private function
        Get Jacoian

        Args:
            x_input: (tf_tensor) Input data

        Returns:
            jaconbian: (tf_tensor) Jacobian
        """

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)

        jacobian = g.batch_jacobian(logits, x_input)

        return logits, jacobian


    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_xent(self, x_input, y_input):
        """ Private function
        Get gradient of cross entropy

        Args:
            x_input: (tf_tensor) Input data
            y_input: (tf_tensor) Input label

        Returns:
            logits: (tf_tensor) Logits
            xent: (tf_tensor) Cross entropy
            grad_xent: (tf_tensor) Gradient of cross entropy
        """

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            xent = self.__get_xent(logits, y_input)
        
        grad_xent = g.gradient(xent, x_input)

        return logits, xent, grad_xent


    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_diff_logits_target(self, x, la, la_target):
        """ Private function
        Get difference of logits and corrospopnding gradient

        Args:
            x_input: (tf_tensor) Input data
            la: (tf_tensor) Input label
            la_target: (tf_tensor) Input targeted label

        Returns:
            difflogits: (tf_tensor) Difference of logits
            grad_diff: (tf_tensor) Gradient of difference of logits
        """

        la_mask = tf.one_hot(la, self.num_classes)
        la_target_mask = tf.one_hot(la_target, self.num_classes)

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x)
            logits = self.__get_logits(x)
            difflogits = tf.reduce_sum((la_target_mask - la_mask) * logits, axis=1)

        grad_diff = g.gradient(difflogits, x)

        return difflogits, grad_diff


    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_dlr(self, x_input, y_input):
        """ Private function
        Get gradient of DLR loss

        Args:
            x_input: (tf_tensor) Input data
            y_input: (tf_tensor) Input label

        Returns:
            logits: (tf_tensor) Logits
            val_dlr: (tf_tensor) DLR loss
            grad_dlr: (tf_tensor) Gradient of DLR loss
        """

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            val_dlr = self.__get_dlr(logits, y_input)

        grad_dlr = g.gradient(val_dlr, x_input)
        
        return logits, val_dlr, grad_dlr


    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __get_grad_dlr_target(self, x_input, y_input, y_target):
        """ Private function
        Get gradient of targeted DLR loss

        Args:
            x_input: (tf_tensor) Input data
            y_input: (tf_tensor) Input label
            y_target: (tf_tensor) Input targeted label

        Returns:
            logits: (tf_tensor) Logits
            val_dlr: (tf_tensor) Targeted DLR loss
            grad_dlr: (tf_tensor) Gradient of targeted DLR loss
        """

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x_input)
            logits = self.__get_logits(x_input)
            dlr_target = self.__get_dlr_target(logits, y_input, y_target)

        grad_target = g.gradient(dlr_target, x_input)

        return logits, dlr_target, grad_target
    

    # Public API #
    def predict(self, x):
        """
        Get model's pre-softmax output in inference mode

        Args:
            x_input: (pytorch_tensor) Input data

        Returns:
            y: (pytorch_tensor) Pre-softmax output
        """

        # Convert pt_tensor to tf format
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        # Get result
        y = self.__get_logits(x2)

        # Convert result to pt format
        y = self.__tf_to_pt(y)
        
        return y


    def grad_logits(self, x):
        """
        Get logits and gradient of logits

        Args:
            x: (pytorch_tensor) Input data

        Returns:
            logits: (pytorch_tensor) Logits
            g2: (pytorch_tensor) Jacobian
        """

        # Convert pt_tensor to tf format
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])
        
        # Get result
        logits, g2 = self.__get_jacobian(x2)

        # Convert result to pt format
        if self.data_format == 'channels_last':
            g2 = tf.transpose(g2, perm=[0,1,4,2,3])
        logits = self.__tf_to_pt(logits)
        g2 = self.__tf_to_pt(g2)

        return logits, g2


    def get_logits_loss_grad_xent(self, x, y):
        """
        Get gradient of cross entropy

        Args:
            x: (pytorch_tensor) Input data
            y: (pytorch_tensor) Input label

        Returns:
            logits_val: (pytorch_tensor) Logits
            loss_indiv_val: (pytorch_tensor) Cross entropy
            grad_val: (pytorch_tensor) Gradient of cross entropy
        """

        # Convert pt_tensor to tf format
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        # Get result
        logits_val, loss_indiv_val, grad_val = self.__get_grad_xent(x2, y2)

        # Convert result to pt format
        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        logits_val = self.__tf_to_pt(logits_val)
        loss_indiv_val = self.__tf_to_pt(loss_indiv_val)
        grad_val = self.__tf_to_pt(grad_val)

        return logits_val, loss_indiv_val, grad_val


    def set_target_class(self, y, y_target):
        pass
    

    def get_grad_diff_logits_target(self, x, y, y_target):
        """
        Get difference of logits and corrospopnding gradient

        Args:
            x: (pytorch_tensor) Input data
            y: (pytorch_tensor) Input label
            y_target: (pytorch_tensor) Input targeted label

        Returns:
            difflogits: (pytorch_tensor) Difference of logits
            g2: (pytorch_tensor) Gradient of difference of logits
        """

        # Convert pt_tensor to tf format
        la = tf.convert_to_tensor(y.cpu().numpy(), dtype=tf.int32)
        la_target = tf.convert_to_tensor(y_target.cpu().numpy(), dtype=tf.int32)
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        # Get result
        difflogits, g2 = self.__get_grad_diff_logits_target(x2, la, la_target)

        # Convert result to pt format
        if self.data_format == 'channels_last':
            g2 = tf.transpose(g2, perm=[0, 3, 1, 2])
        difflogits = self.__tf_to_pt(difflogits)
        g2 = self.__tf_to_pt(g2)
        
        return difflogits, g2


    def get_logits_loss_grad_dlr(self, x, y):
        """
        Get gradient of DLR loss

        Args:
            x: (pytorch_tensor) Input data
            y: (pytorch_tensor) Input label

        Returns:
            logits_val: (pytorch_tensor) Logits
            loss_indiv_val: (pytorch_tensor) DLR loss
            grad_val: (pytorch_tensor) Gradient of DLR loss
        """

        # Convert pt_tensor to tf format
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        # Get result
        logits_val, loss_indiv_val, grad_val = self.__get_grad_dlr(x2, y2)

        # Convert result to pt format
        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        logits_val = self.__tf_to_pt(logits_val)
        loss_indiv_val = self.__tf_to_pt(loss_indiv_val)
        grad_val = self.__tf_to_pt(grad_val)

        return logits_val, loss_indiv_val, grad_val
    
    def get_logits_loss_grad_target(self, x, y, y_target):
        """
        Get gradient of targeted DLR loss

        Args:
            x: (pytorch_tensor) Input data
            y: (pytorch_tensor) Input label
            y_target: (pytorch_tensor) Input targeted label

        Returns:
            logits_val: (pytorch_tensor) Logits
            loss_indiv_val: (pytorch_tensor) Targeted DLR loss
            grad_val: (pytorch_tensor) Gradient of targeted DLR loss
        """

        # Convert pt_tensor to tf format
        x2 = tf.convert_to_tensor(x.cpu().numpy(), dtype=tf.float32)
        y2 = tf.convert_to_tensor(y.cpu().numpy(), dtype=tf.int32)
        y_targ = tf.convert_to_tensor(y_target.cpu().numpy(), dtype=tf.int32)
        if self.data_format == 'channels_last':
            x2 = tf.transpose(x2, perm=[0,2,3,1])

        # Get result
        logits_val, loss_indiv_val, grad_val = self.__get_grad_dlr_target(x2, y2, y_targ)

        # Convert result to pt format
        if self.data_format == 'channels_last':
            grad_val = tf.transpose(grad_val, perm=[0,3,1,2])
        logits_val = self.__tf_to_pt(logits_val)
        loss_indiv_val = self.__tf_to_pt(loss_indiv_val)
        grad_val = self.__tf_to_pt(grad_val)

        return logits_val, loss_indiv_val, grad_val
