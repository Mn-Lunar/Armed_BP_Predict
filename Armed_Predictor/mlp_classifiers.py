"""Simple neural networks for classification
"""
import tensorflow as tf
import tensorflow.keras.layers as tkl
from .random_effects import RandomEffects

class BaseMLP(tf.keras.Model):
    # 使用TensorFlow 的 keras 框架
    # 
    def __init__(self, name: str='mlp', **kwargs):
        """Basic MLP with 3 hidden layers of 4 neurons each.

        Args:
            name (str, optional): Model name. Defaults to 'mlp'.
        """        
        super(BaseMLP, self).__init__(name=name, **kwargs)  
        # **kwargs = 额外参数（可选）

        # ** NEW **
        # 新增一层 flatten 进行展平
        # (batch, 1250, 1) → (batch, 1250)
        # 转移任务至trainer中，在trainer的_process_input()方法中进行数据格式修改
        #self.flatten = tf.keras.layers.Flatten(name=name + '_flatten')
        

        #归一化层
        self.norm = tkl.LayerNormalization(name=name + '_norm')

        # dense: 线性层（全连接层）
        # 基础结构 
        self.dense0 = tkl.Dense(128, activation='relu', name=name + '_dense0')
        self.dense1 = tkl.Dense(64, activation='relu', name=name + '_dense1')
        self.dense2 = tkl.Dense(32, activation='relu', name=name + '_dense2')
        # ** NEW **
        # 更改为线性输出 以 适配回归任务
        self.dense_out = tkl.Dense(1, activation='linear', name=name + '_dense_out')
        
    def call(self, inputs):
        
        x = self.norm(inputs)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_out(x)
        
        return x
    
class ClusterCovariateMLP(BaseMLP):
    """
    Basic MLP that concatenates the site membership design matrix to the data.
    """
    def call(self, inputs):
        x, z = inputs
        
        x = tf.concat((x, z), axis=1)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_out(x)
    
        return x
    

class MLPActivations(tkl.Layer):
    def __init__(self, last_activation: str='sigmoid', name: str='mlp_activations', **kwargs):
        """Basic MLP with 3 hidden layers of 4 neurons each. In addition to the
        prediction, also returns the activation of each layer. Intended to be
        used within a domain adversarial model.

        Args: 
        last_activation (str, optional): Activation of output layer. Defaults to 
            'sigmoid'. 
        name (str, optional): Model name. Defaults to 'mlp_activations'.
        """        
        super(MLPActivations, self).__init__(name=name, **kwargs)

        self.dense0 = tkl.Dense(4, activation='relu', name=name + '_dense0')
        self.dense1 = tkl.Dense(4, activation='relu', name=name + '_dense1')
        self.dense2 = tkl.Dense(4, activation='relu', name=name + '_dense2')
        self.dense_out = tkl.Dense(1, activation=last_activation, name=name + '_dense_out')
        
    def call(self, inputs):
        
        x0 = self.dense0(inputs)
        x1 = self.dense1(x0)
        x2 = self.dense2(x1)
        out = self.dense_out(x2)
        
        return x0, x1, x2, out
    
    def get_config(self):
        return {}
    
class Adversary(tkl.Layer):
    def __init__(self,
                 n_clusters: int, 
                 layer_units: list=[8, 8, 4],
                 name: str='adversary',
                 **kwargs):
        """Adversarial classifier. 

        Args:
            n_clusters (int): number of clusters (classes)
            layer_units (list, optional): Neurons in each layer. Can be a list of any
                length. Defaults to [8, 8, 8].
            name (str, optional): Model name. Defaults to 'adversary'.
        """        
        
        super(Adversary, self).__init__(name=name, **kwargs)
        
        self.n_clusters = n_clusters
        self.layer_units = layer_units
        
        # ** 构建神经网络结构 **
        # enumerate(layer_units) 是一个 p迭代器， 第一个参数是 index 第二个参数是数组中的值
        # 对于 layer_units: list=[8, 8, 4]
        # 第一次循环 iLayer = 0 neurons = 8
        # 第二次循环 iLayer = 1 neurons = 8
        # 第三次循环 iLayer = 2 neurons = 4
        self.layers = []
        for iLayer, neurons in enumerate(layer_units):
            # 每次一添加一个隐藏层， 参数为 neurons 数组中对应的神经元数，每个都是 relu函数，名字是 < _dense + 层数 >
            self.layers += [tkl.Dense(neurons, 
                                      activation='relu', 
                                      name=name + '_dense' + str(iLayer))]
        # 最后 添加一个 softmax 函数 作为输出层， 得到包含每一簇的概率 h hat 数组   
        self.layers += [tkl.Dense(n_clusters, activation='softmax', name=name + '_dense_out')]

    # 前向传播
    # 这里就 逐层计算每个隐藏层然后返回最终的输出结果   
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            # 调用layer.__call__(x)
            # __call__() 调用 call(x)
            # 总之就是计算这一层神经元的函数然后输出一个新的 x           
        return x
    
    # 简单的get函数得到：
    # 1. 该网络 簇的个数 'n_clusters': self.n_clusters
    # 2. 该网络 每一层的神经元数量 （例： layer_units: list=[8, 8, 4]） 'layer_units': self.layer_units
    def get_config(self):
        return {'n_clusters': self.n_clusters,
                'layer_units': self.layer_units}
        
class DomainAdversarialMLP(tf.keras.Model):
    def __init__(self, 
                 n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 name: str='da_mlp', 
                 **kwargs):
        """Domain adversarial MLP classifier. The main model learns the classification
        task while the adversary prevents it from learning cluster-related features. 

        Args:
            n_clusters (int): Number of clusters. 簇的个数
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4]. 对抗网络每一层的神经元数目
            name (str, optional): Model name. Defaults to 'da_mlp'.
        """        
        
        super(DomainAdversarialMLP, self).__init__(name=name, **kwargs)

        # 网络 1： MLP Activations
        self.classifier = MLPActivations(name='mlp')
        # 网络 2： Adversary
        self.adversary = Adversary(n_clusters=n_clusters, 
                                   layer_units=adversary_layer_units,
                                   name='adversary')
        
    def call(self, inputs):
        x, z = inputs
        # x 是训练数据的特征值
        # z 是簇相关的向量

        # 先用 fixed effect 网络对输入值进行前向传播
        # 返回一个 列表 classifier_outs， 包含每一层的输出值
        # Eg: classifier_outs = [hidden1, hidden2, hidden3, output]
        classifier_outs = self.classifier(x)
        
        # 取最后一层输出做pred_class，也就是预测结果 h hat
        pred_class = classifier_outs[-1] 
        
        # 取出前三层的输出值输出给对抗网络
        activations = tf.concat(classifier_outs[:3], axis=1) 
        pred_cluster = self.adversary(activations)
        
        return pred_class, pred_cluster
    
    def compile(self,
                loss_class=tf.keras.losses.BinaryCrossentropy(), # fixed effect：主分类任务，一个适用于二分类任务的损失函数
                loss_adv=tf.keras.losses.CategoricalCrossentropy(), # random effect： 对抗网络部分的损失函数
                metric_class=tf.keras.metrics.AUC(curve='PR', name='auprc'), # 评估指标部分 
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_main=tf.keras.optimizers.Adam(lr=0.001), # 优化器部分
                opt_adversary=tf.keras.optimizers.Adam(lr=0.001),
                loss_class_weight=1.0, #
                loss_gen_weight=1.0,
                ):
        """Compile model with selected losses and metrics. Must be called before training.
        
        Loss weights apply to the main model: 
        total_loss = loss_class_weight * loss_class - loss_gen_weight * loss_adv

        Args:
            loss_class (loss, optional): Main classification loss. Defaults to 
                tf.keras.losses.BinaryCrossentropy().
            loss_adv (loss, optional): Adversary classification loss. Defaults to 
                tf.keras.losses.CategoricalCrossentropy().
            metric_class (metric, optional): Main classification metric. Defaults to 
                tf.keras.metrics.AUC(curve='PR', name='auprc').
            metric_adv (metric, optional): Adversary classification metric. Defaults to 
                tf.keras.metrics.CategoricalAccuracy(name='acc').
            opt_main (optimizer, optional): Main optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            opt_adversary (optimizer, optional): Adversary optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            loss_class_weight (float, optional): Classification loss weight. Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. Defaults to 1.0.
        """        
        
        super().compile()
        
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_main = opt_main
        self.opt_adversary = opt_adversary
        
        # Trackers for running mean of each loss
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class = metric_class
        self.metric_adv = metric_adv

        self.loss_class_weight = loss_class_weight
        self.loss_gen_weight = loss_gen_weight    
        
    @property
    def metrics(self):
        return [self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv]
        
    def train_step(self, data): # 这个data是 TensorFlow 中 .fit() 输入的数据
        # Unpack data, including sample weights if provided
        if len(data) == 3:
            (data, clusters), labels, sample_weights = data
            # [(（x, z）, y ) , ....] 
            # x 为fixed effect训练的数据
            # z 为mixed effect训练的数据
            # y x训练时需要的标签
            # ... 为其它参数
        else:
            (data, clusters), labels = data
            sample_weights = None
        
        ##############################################################################################
        # Get hidden layer activations from classifier and train the adversary 
        # *这一层的逻辑：fixed effect中可能使用了mixed effect的信息作为特征提取，每层输出可能都带有相关的内容
        # *于是我们将隐藏层的输出提供给 adversary 网络， 让它从相关特征信息中预测 cluster   
        
        activations = tf.concat(self.classifier(data)[:-1], axis=1)
        # 得到当前 fixed effect classifier 的除去最后一层的数据
        # x0, x1, x2, out = dense0 → dense1 → dense2 → dense_out
        # 这里也就是提取隐藏层的输出 x0, x1, x2 
        # tf.concat( x0, x1, x2 ..., axis=1) 这个方法将每层数据按列拼接
        # 也就是 [1 2 3], [4 5 6], [7 8 9] -> [1 2 3 4 5 6 7 8 9] 

        with tf.GradientTape() as gt:
            # 对 adversary 前向传播 得到 cluster 预测值
            pred_cluster = self.adversary(activations) # 之前提取到的 fixed effect classifier 中的隐藏层输出，用来提供给adversatry网络作为输入
            
            # 具体不明，反正就是一个损失函数用来比较 cluster 的 真实值和预测值
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)

        # 更新损失函数的梯度    
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables) # 对损失函数求可训练参数的梯度
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables)) # 优化参数
        
        # 更新可视化指标
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        

        ##############################################################################################
        # Train the main classifier
        # Fixed effect classifier 的训练步骤，它会试图“欺骗”adversary 网络，让隐藏层不含 cluster 信息。
        with tf.GradientTape() as gt2:
            
            # 参数： 类别（fixed effect主要任务）  簇信息（mixed effect 次要任务）
            pred_class, pred_cluster = self((data, clusters))# ！！！这里调用的函数里使用了adversary网络 得到 pred cluster
            # 主要任务（分辨样本所在的类）相关的损失函数
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            # 次要任务（和adversary网络一致的损失函数）
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            # 最大化 主要任务的 损失函数， 同时最小化 次要任务的损失函数，从而减少 cluster 相关的特征值
            # 人话： 降低 mixed effect 造成的影响， 从而使adversary网络变得更难预测 cluster 相关的信息
            total_loss = (self.loss_class_weight * loss_class) \
                - (self.loss_gen_weight * loss_adv)
    
        # 梯度更新
        grads_class = gt2.gradient(total_loss, self.classifier.trainable_variables)
        self.opt_main.apply_gradients(zip(grads_class, self.classifier.trainable_variables))
        

        self.metric_class.update_state(labels, pred_class)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (data, clusters), labels = data
                        
        pred_class, pred_cluster = self((data, clusters))
        loss_class = self.loss_class(labels, pred_class)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_class_weight * loss_class) \
            - (self.loss_gen_weight * loss_adv)
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_tracker.update_state(loss_class)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    

class RandomEffectsLinearSlopeIntercept(tkl.Layer):
    def __init__(self, 
                 slopes: int,
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001, 
                 name: str='randomeffects', **kwargs):
        """Layer that learns a random linear slope and intercept. When called on an input
        (x, z), it returns a tuple of (f(random_slope(z) * x), random_intercept(z)).

        Args:
            slopes ([type]): dimensionality of the slopes (i.e. the number of features)
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Mode name. Defaults to 'randomeffects'.
        """        
        super(RandomEffectsLinearSlopeIntercept, self).__init__(name=name, **kwargs)
    
        self.slopes = slopes
        self.slope_posterior_init_scale = slope_posterior_init_scale
        self.intercept_posterior_init_scale = intercept_posterior_init_scale
        self.slope_prior_scale = slope_prior_scale
        self.intercept_prior_scale = intercept_prior_scale
        self.kl_weight= kl_weight
        
        self.re_slope = RandomEffects(slopes, 
                                      post_loc_init_scale=slope_posterior_init_scale,
                                      prior_scale=slope_prior_scale,
                                      kl_weight=kl_weight, name=name + '_re_slope')
        self.dense_out = tkl.Dense(1, name=name + '_re_out')
        
        self.re_int = RandomEffects(1, 
                                    post_loc_init_scale=intercept_posterior_init_scale,
                                    prior_scale=intercept_prior_scale,
                                    kl_weight=kl_weight, 
                                    name=name + '_re_int')
  
    def call(self, inputs, training=None):
        x, z = inputs        
        slope = self.re_slope(z, training=training)
        # prod = self.dense_out(x * slope)
        prod = tf.reduce_sum(x * slope, axis=1, keepdims=True)
        intercept = self.re_int(z, training=training)
        
        return  prod, intercept
    
    def get_config(self):
        return {'slopes': self.slopes,
                'slope_posterior_init_scale': self.slope_posterior_init_scale,
                'intercept_posterior_init_scale': self.intercept_posterior_init_scale,
                'slope_prior_scale': self.slope_prior_scale,
                'intercept_prior_scale': self.intercept_prior_scale,
                'kl_weight': self.kl_weight}
        
class MixedEffectsMLP(DomainAdversarialMLP):
    def __init__(self, n_features: int, n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001,
                 name: str='me_mlp', 
                 **kwargs):
        """Mixed effects MLP classifier. Includes an adversarial classifier to 
        disentangle the predictive features from the cluster-specific features. 
        The cluster-specific features are then learned by a random effects layer. 
        
        This architecture includes linear random slopes (to be multiplied by the 
        input) and random intercept. The model output is 
        (fixed effect output) + (random slopes) * X + (random intercept)

        Args:
            n_features (int): Number of features.
            n_clusters (int): Number of clusters.
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4].
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Model name. Defaults to 'me_mlp'.
        """        
    
        super(MixedEffectsMLP, self).__init__(n_clusters=n_clusters,
                                              adversary_layer_units=adversary_layer_units,
                                              name=name, **kwargs)
        self.classifier = MLPActivations(last_activation='linear', name='mlp')

        self.randomeffects = RandomEffectsLinearSlopeIntercept(
                        n_features,
                        slope_posterior_init_scale=slope_posterior_init_scale,
                        intercept_posterior_init_scale=intercept_posterior_init_scale,
                        slope_prior_scale=slope_prior_scale,
                        intercept_prior_scale=intercept_prior_scale,
                        kl_weight=kl_weight)

        
    def call(self, inputs, training=None):
        x, z = inputs
        # 对最开始的普通 MLP 网络进行前向传播
        # 取得每一层的输出
        fe_outs = self.classifier(x)
        pred_class_fe = tf.nn.sigmoid(fe_outs[-1]) # 取最后一层输入sigmoid函数
        
        #################################
        ############# 关键点 #############
        # 获取slope 和 intercept，用来微调每个cluster        
        re_prod, re_int = self.randomeffects((x, z), training=training)
        pred_class_me = tf.nn.sigmoid(re_prod + re_int + fe_outs[-1])     
        
        
        #对抗网络的部分，取出前三层给adversary网络做输入，和之前一致
        fe_activations = tf.concat(fe_outs[:3], axis=1)
        pred_cluster = self.adversary(fe_activations)
                
        return pred_class_me, pred_class_fe, pred_cluster
    
    def compile(self,
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.CategoricalCrossentropy(),
                metric_class_me=tf.keras.metrics.AUC(curve='PR', name='auprc'),
                metric_class_fe=tf.keras.metrics.AUC(curve='PR', name='auprc_fe'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_main=tf.keras.optimizers.Adam(lr=0.001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.001),
                loss_class_me_weight=1.0,
                loss_class_fe_weight=1.0,
                loss_gen_weight=1.0,
                ):
        """Compile model with selected losses and metrics. Must be called before training.
        
        Loss weights apply to the main model: 
        total_loss = loss_class_me_weight * loss_class_me + loss_class_fe_weight * loss_class_fe
            - loss_gen_weight * loss_adv

        Args:
            loss_class (loss, optional): Main classification loss. This applies to both the 
                mixed and fixed effects-based classifications. Defaults to 
                tf.keras.losses.BinaryCrossentropy().
            loss_adv (loss, optional): Adversary classification loss. Defaults to 
                tf.keras.losses.CategoricalCrossentropy().
            metric_class_me (metric, optional): Metric for classification using mixed effects. 
                Defaults to tf.keras.metrics.AUC(curve='PR', name='auprc').
            metric_class_fe (metric, optional): Metric for classification using fixed effects. 
                Defaults to tf.keras.metrics.AUC(curve='PR', name='auprc_fe').
            metric_adv (metric, optional): Adversary classification metric. Defaults to 
                tf.keras.metrics.CategoricalAccuracy(name='acc').
            opt_main (optimizer, optional): Main optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            opt_adversary (optimizer, optional): Adversary optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            loss_class_me_weight (float, optional): Weight for classification using mixed 
                effects. Defaults to 1.0.
            loss_class_fe_weight (float, optional): Weight for classification using fixed 
                effects. Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. Defaults to 1.0.
        """  
        
        super().compile()
        
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_main = opt_main
        self.opt_adversary = opt_adversary
        
        # Loss trackers
        self.loss_class_me_tracker = tf.keras.metrics.Mean(name='class_me_loss')
        self.loss_class_fe_tracker = tf.keras.metrics.Mean(name='class_fe_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class_me = metric_class_me
        self.metric_class_fe = metric_class_fe
        self.metric_adv = metric_adv

        self.loss_class_me_weight = loss_class_me_weight
        self.loss_class_fe_weight = loss_class_fe_weight
        self.loss_gen_weight = loss_gen_weight    
        
        # Unneeded
        del self.loss_class_tracker, self.loss_class_weight, self.metric_class
        
    @property
    def metrics(self):
        return [self.loss_class_me_tracker,
                self.loss_class_fe_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class_me,
                self.metric_class_fe,
                self.metric_adv]
        
    def train_step(self, data):
        # Unpack data, including sample weights if provided
        if len(data) == 3:
            (data, clusters), labels, sample_weights = data
        else:
            (data, clusters), labels = data
            sample_weights = None
        
        # Get hidden layer activations from classifier and train the adversary       
        activations = tf.concat(self.classifier(data)[:-1], axis=1)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(activations)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        # Train the main classifier 
        with tf.GradientTape() as gt2:
            pred_class_me, pred_class_fe, pred_cluster = self((data, clusters), training=True)
            loss_class_me = self.loss_class(labels, pred_class_me, sample_weight=sample_weights)
            loss_class_fe = self.loss_class(labels, pred_class_fe, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            total_loss = (self.loss_class_me_weight * loss_class_me) \
                + (self.loss_class_fe_weight * loss_class_fe) \
                - (self.loss_gen_weight * loss_adv) \
                + self.randomeffects.losses

        lsVars = self.classifier.trainable_variables + self.randomeffects.trainable_variables
        grads_class = gt2.gradient(total_loss, lsVars)
        self.opt_main.apply_gradients(zip(grads_class, lsVars))
        
        self.metric_class_me.update_state(labels, pred_class_me)
        self.metric_class_fe.update_state(labels, pred_class_fe)
        self.loss_class_me_tracker.update_state(loss_class_me)
        self.loss_class_fe_tracker.update_state(loss_class_fe)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (data, clusters), labels = data
                        
        pred_class_me, pred_class_fe, pred_cluster = self((data, clusters), training=False)
        loss_class_me = self.loss_class(labels, pred_class_me)
        loss_class_fe = self.loss_class(labels, pred_class_fe)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_class_me_weight * loss_class_me) \
                + (self.loss_class_fe_weight * loss_class_fe) \
                - (self.loss_gen_weight * loss_adv) \
                + self.randomeffects.losses
                    
        self.metric_class_me.update_state(labels, pred_class_me)
        self.metric_class_fe.update_state(labels, pred_class_fe)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_me_tracker.update_state(loss_class_me)
        self.loss_class_fe_tracker.update_state(loss_class_fe)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
        
        
class MixedEffectsMLPNonlinearSlope(MixedEffectsMLP):
    def __init__(self, n_features: int, n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001,
                 name: str='me_mlp', 
                 **kwargs):
        """Mixed effects MLP classifier. Includes an adversarial classifier to 
        disentangle the predictive features from the cluster-specific features. 
        The cluster-specific features are then learned by a random effects layer. 

        This architecture includes nonlinear random slopes (to be multiplied by the 
        penultimate layer output of the fixed effects submodel) and random intercept. 
        The model output is 
        (fixed effect output) + (random slopes) * (penultimate FE layer output) + (random intercept)

        Args:
            n_features (int): Number of features.
            n_clusters (int): Number of clusters.
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4].
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Model name. Defaults to 'me_mlp'.
        """       
        del n_features # unused
    
        super(MixedEffectsMLP, self).__init__(n_clusters=n_clusters,
                                              adversary_layer_units=adversary_layer_units,
                                              name=name, **kwargs)
        self.classifier = MLPActivations(last_activation='linear', name='mlp')

        self.randomeffects = RandomEffectsLinearSlopeIntercept(
                        slopes=4,
                        slope_posterior_init_scale=slope_posterior_init_scale,
                        intercept_posterior_init_scale=intercept_posterior_init_scale,
                        slope_prior_scale=slope_prior_scale,
                        intercept_prior_scale=intercept_prior_scale,
                        kl_weight=kl_weight)

    def call(self, inputs, training=None):
        x, z = inputs
        fe_outs = self.classifier(x)
        pred_class_fe = tf.nn.sigmoid(fe_outs[-1])
        
        # Penultimate FE layer output
        fe_latents = fe_outs[-2]        
        
        re_prod, re_int = self.randomeffects((fe_latents, z), training=training)
        pred_class_me = tf.nn.sigmoid(re_prod + re_int + pred_class_fe)     
        
        fe_activations = tf.concat(fe_outs[:3], axis=1)
        pred_cluster = self.adversary(fe_activations)
                
        return pred_class_me, pred_class_fe, pred_cluster