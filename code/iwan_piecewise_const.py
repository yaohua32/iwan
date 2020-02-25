# -*- coding: utf-8 -*-
class visual_():
    
    def __init__(self, file_path):
        self.dir= file_path
        
    def show_error(self, iteration, error, name1, name2, dim):
        # 画 L_2 relative error vs. iteration 图像的函数
        # for drawing L_2 relative error vs. iteration
        plt.figure(figsize=(8,7))
        plt.semilogy(iteration, error, color='b')
        plt.xlabel("Iteration", size=28)
        plt.ylabel(name1, size=28)        
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_iter_%s_%dd.png'%(name2, dim))
        plt.close()
        
    def show_error_abs(self, mesh, x_y, z, name, dim):
        # 画pointwise absolute error 图像的函数
        # for drawing point-wise absolute error
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure(figsize=(8,7)) 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=28)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=28)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_abs_%s_%dd.png'%(name, dim))
        plt.close()
        
    def show_u_val(self, mesh, x_y, z1, z2, name, num):
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        #*******************
        fig= plt.figure(figsize=(12,5))
        ax1= fig.add_subplot(1,2,1)
        z1i = griddata((x, y), np.ravel(z1), (xi, yi), method='linear')
        graph1= plt.contourf(xi, yi, z1i, 15, cmap=plt.cm.jet)
        fig.colorbar(graph1, ax= ax1)
        #
        ax2= fig.add_subplot(1,2,2)
        z2i= griddata((x, y), np.ravel(z2), (xi, yi), method='linear')
        graph2= ax2.contourf(xi, yi, z2i, 15,  cmap= cm.jet)
        fig.colorbar(graph2, ax= ax2)
        #*******************
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/iwan_%s_%d.png'%(name, name, num))
        plt.close()
        
    def show_v_val(self, mesh, x_y, z, name, num):
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure(figsize=(8,7)) 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=28)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=28)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/iwan_%s_%d.png'%(name, name, num))
        plt.close()
        
class wan_inv():
    
    def __init__(self, dim, noise_level, dm_size, bd_size, beta_u, beta_bd,
                 u_step, u_rate, v_step, v_rate, file_path, iteration):
        import numpy as np
        global np
        #
        import time
        global time
        #
        import tensorflow as tf
        global tf
        #
        import matplotlib.pyplot as plt
        global plt
        #
        from scipy.interpolate import griddata
        global griddata
        #
        from scipy.stats import truncnorm
        global truncnorm
        # 
        from matplotlib import cm
        global cm
        #
        self.dim= dim                            #问题的维度
        self.noise_level= noise_level
        self.up, self.low=   1.0, -1.0           #矩形区域[-1,1]^d
        self.k=   [0.81, 2.0]+[0.09]*(dim-2)     #\omega_1区域表达式中:前面的系数
        self.c_a= [0.1, 0.3]+[0.0]*(dim-2)       #\omega_1区域表达式中:区域的中心
        self.c_u= [0, 0]+[0.0]*(dim-2)           #真实解表达式中:最小值点
        self.r= 0.6                              #\omega_1区域表达式中：半径值
        self.alpha= 0.02                          #用来控制不连续程度的值（越小奇异性越大）
        self.a1= 2.0                             #coefficient a(x) 在\omega_1区域内的值
        self.a2= 0.5                             #coefficient a(x)  在\omega_1区域之外的值
        self.mesh_size= 100                      #用来生成testing data
        self.beta_u= beta_u                      #loss function for boundary of u(x) 前面的参数
        self.beta_bd= beta_bd
        #
        self.v_layer= 6                          #test function v  的hidden layers 层数
        self.v_h_size= 20                        #test function v  每层的neuron 数目
        #
        self.a_layer= 4                         
        self.a_h_size= 20
        self.u_layer= 6                          
        self.u_h_size= 20 
        #                       
        self.u_step= u_step                           #解u(x)内循环（神经网络u的迭代步数）
        self.u_rate= u_rate                       #解u(x)内循环（神经网络u的learning rate）
        self.v_step_u= v_step                         #解u(x)内循环（test function v的迭代步数）
        self.v_rate_u= v_rate                     #解u(x)内循环（test function v的learning rate）
        #
        self.dm_size= dm_size                       #内部采样点数目                   
        self.bd_size= bd_size                       #边界采样点数目
        self.iteration= iteration
        #
        self.dir= file_path              #运行的时候需要建一个文件夹，以此名字命名，然后在该文件夹下面
                                                 #新建文件夹figure_err, figure_u, figure_a, figure_v，分别用来保存中间过程输出的图像
        
    def get_truncated_normal(self, mean=0.0, sd=1.0):
        # 观测噪音生成函数
        #for adding noise
        low= -100; up= 100
        result= truncnorm((low-mean)/sd, (up-mean)/sd, loc=mean, scale=sd)
        return(result)
        
    def sample_train(self, dm_size, bd_size, dim):
        # 生成训练数据
        low, up= self.low, self.up
        distb= self.get_truncated_normal()
        #********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
        #*********************************************************
        # The value of f(x)
        omega_a, omega_u= 0.0, 0.0
        for i in range(dim):
            omega_a= omega_a+self.k[i]**2*(x_dm[:,i]-self.c_a[i])**2
            omega_u= omega_u+self.k[i]**2*(x_dm[:,i]-self.c_a[i])*(x_dm[:,i]-self.c_u[i])
        exp_term= np.exp((omega_a-self.r**2)/self.alpha)
        #
        part_one= 4*(self.a1-self.a2)*omega_u/(self.alpha/exp_term+2*self.alpha+self.alpha*exp_term)
        part_two= 2*dim*(self.a2*(1-1/(1+exp_term))+self.a1/(1+exp_term))
        f_dm= part_one-part_two
        f_dm= np.reshape(f_dm, [-1,1])
        #*********************************************************
        # collocation points on boundary
        x_bd_list=[]
        n_vector_list=[]
        for i in range(dim):
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            x_bound[:,i]= up
            x_bd_list.append(x_bound)
            n_vector= np.zeros_like(x_bound)
            n_vector[:,i]=1
            n_vector_list.append(n_vector)
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            x_bound[:,i]= low
            x_bd_list.append(x_bound)
            n_vector= np.zeros_like(x_bound)
            n_vector[:,i]=-1
            n_vector_list.append(n_vector)
        x_bd= np.concatenate(x_bd_list, axis=0)
        n_vector= np.concatenate(n_vector_list, 0)
        #***********************************************************
        # observation of u(x) on boundary
        u_bd= 0.0
        for i in range(dim):
            u_bd= u_bd+(x_bd[:,i]-self.c_u[i])**2
        u_bd= np.reshape(u_bd, [-1, 1])
        #*********************************************************
        # observation of a(x) on boundary
        omega_a_bd= 0.0
        for i in range(dim):
            omega_a_bd= omega_a_bd+self.k[i]**2*(x_bd[:,i]-self.c_a[i])**2
        exp_term_bd= np.exp((omega_a_bd-self.r**2)/self.alpha)
        #
        a_bd= (self.a2*(1-1/(1+exp_term_bd))+self.a1/(1+exp_term_bd))
        a_bd= np.reshape(a_bd, [-1,1])
        #********************************************************
        train_dict={}
        x_dm= np.float32(x_dm); train_dict['x_dm']= x_dm
        f_dm= np.float32(f_dm); train_dict['f_dm']= f_dm
        x_bd= np.float32(x_bd); train_dict['x_bd']= x_bd
        u_bd= np.float32(u_bd); train_dict['u_bd']= u_bd
        a_bd= np.float32(a_bd); train_dict['a_bd']= a_bd
        n_vector= np.float32(n_vector); train_dict['n_vector']=n_vector
        return(train_dict)
        
    def sample_test(self, mesh_size, dim):
        # 生成测试数据
        low, up= self.low, self.up
        #**********************************************************
        # generate meshgrid in the domain
        x_mesh= np.linspace(low, up, mesh_size)
        mesh= np.meshgrid(x_mesh, x_mesh)
        x1_dm= np.reshape(mesh[0], [-1,1])
        x2_dm= np.reshape(mesh[1], [-1,1])
        #
        x3_dm= np.random.uniform(low, up, [self.mesh_size*self.mesh_size, dim-2])
        x_dm= np.concatenate([x1_dm, x2_dm, x3_dm], axis=1)
        x4_dm= np.zeros([self.mesh_size*self.mesh_size, dim-2])
        x_draw_dm= np.concatenate([x1_dm, x2_dm, x4_dm], axis=1)
        #***********************************************************
        # The exact u(x)
        u_dm= 0.0
        u_draw_dm= 0.0
        for i in range(dim):
            u_dm= u_dm+(x_dm[:,i]-self.c_u[i])**2
            u_draw_dm= u_draw_dm+(x_draw_dm[:,i]-self.c_u[i])**2
        u_dm= np.reshape(u_dm, [-1, 1])
        u_draw_dm= np.reshape(u_draw_dm, [-1, 1])
        #***********************************************************
        # The exact a(x)
        omega_a= 0.0
        omega_draw_a= 0.0
        for i in range(dim):
            omega_a= omega_a+self.k[i]**2*(x_dm[:,i]-self.c_a[i])**2
            omega_draw_a= omega_draw_a+self.k[i]**2*(x_draw_dm[:,i]-self.c_a[i])**2
        exp_term= np.exp((omega_a-self.r**2)/self.alpha)
        exp_draw_term= np.exp((omega_draw_a-self.r**2)/self.alpha)
        #
        a_dm= (self.a2*(1-1/(1+exp_term))+self.a1/(1+exp_term))
        a_dm= np.reshape(a_dm, [-1,1])
        a_draw_dm= (self.a2*(1-1/(1+exp_draw_term))+self.a1/(1+exp_draw_term))
        a_draw_dm= np.reshape(a_draw_dm, [-1,1])
        #***********************************************************
        test_dict={}
        test_dict['mesh']= mesh
        x_dm= np.float32(x_dm); test_dict['test_x']= x_dm
        u_dm= np.float32(u_dm); test_dict['test_u']= u_dm
        a_dm= np.float32(a_dm); test_dict['test_a']= a_dm
        x_draw_dm= np.float32(x_draw_dm); test_dict['draw_x']= x_draw_dm
        u_draw_dm= np.float32(u_draw_dm); test_dict['draw_u']= u_draw_dm
        a_draw_dm= np.float32(a_draw_dm); test_dict['draw_a']= a_draw_dm
        return(test_dict)
 
    def net_a(self, x_in, out_size, name, reuse):
        # 逼近 a(x) 的神经网络
        #*****************************************************
        # Neural Net for a(x) (The output should be postive number.)
        h_size= self.a_h_size
        with tf.variable_scope(name, reuse=reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh, name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='input_layer1')
            for i in range(self.a_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.elu, name='h_layer'+str(i))
                else:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='h_layer'+str(i))
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.sigmoid, name='output_layer1')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.sigmoid, name='output_layer2')
            out= tf.layers.dense(hi, out_size, activation= tf.nn.elu, name='output_layer')
        return(out)
    
    def net_u(self, x_in, out_size, name, reuse):
        # 逼近 u(x) 的神经网络
        #*******************************************************
        # Neural Net for u(x)
        h_size= self.u_h_size
        with tf.variable_scope(name, reuse=reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh, name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='input_layer1')
            for i in range(self.u_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus, name= 'h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='output_layer')
        return(out)
        
    def net_v(self, x_in, out_size, name, reuse):
        # 逼近 v(x) 的神经网络
        #*********************************************************
        # Neural Net for v(x)
        h_size= self.v_h_size
        with tf.variable_scope(name, reuse=reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh, name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='input_layer1')
            for i in range(self.v_layer):
                if i%2==0:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='output_layer')
        return(out)
   
    def fun_w(self, x, low, up):
        I1= 0.110987
        x_list= tf.split(x, self.dim, 1)
        #
        x_scale_list=[]
        h_len= (up-low)/2.0
        for i in range(self.dim):
            x_scale= (x_list[i]-low-h_len)/h_len
            x_scale_list.append(x_scale)
        #
        z_x_list=[];
        for i in range(self.dim):
            supp_x= tf.greater(1-tf.abs(x_scale_list[i]), 0)
            z_x= tf.where(supp_x, tf.exp(1/(tf.pow(x_scale_list[i], 2)-1))/I1, 
                          tf.zeros_like(x_scale_list[i]))
            z_x_list.append(z_x)
        #
        w_val= tf.constant(1.0)
        for i in range(self.dim):
            w_val= tf.multiply(w_val, z_x_list[i])
        dw= tf.gradients(w_val, x, unconnected_gradients='zero')[0]
        dw= tf.where(tf.is_nan(dw), tf.zeros_like(dw), dw)
        return(w_val, dw)
    
    def grad_u(self, x_in, name, out_size=1):
        # 计算神经网络u(x)的数值和导数
        u_val= self.net_u(x_in, out_size, name, tf.AUTO_REUSE)
        #
        grad_u= tf.gradients(u_val, x_in, unconnected_gradients='zero')[0]
        return(u_val, grad_u)
        
    def grad_v(self, x_in, name, out_size=1):
        # 计算神经网络v(x)的数值和导数
        v_val= self.net_v(x_in, out_size, name, tf.AUTO_REUSE)
        #
        grad_v= tf.gradients(v_val, x_in, unconnected_gradients='zero')[0]
        return(v_val, grad_v)

    def fun_g(self, x, n_vec):
        x_list= tf.split(x, self.dim, 1)
        #**************************************
        u_val= tf.zeros_like(x_list[0])
        for i in range(self.dim):
            u_val= tf.add(u_val, tf.pow(x_list[i]-self.c_u[i], 2))
        u_val= tf.reshape(u_val, [-1,1])
        #
        du= tf.gradients(u_val, x, unconnected_gradients='zero')[0]
        g_obv= tf.reduce_sum(tf.multiply(du, n_vec), axis=1)
        g_obv= tf.reshape(g_obv, [-1,1])
        return(u_val, du, g_obv)
    
    def build(self):
        #*********************************************************************
        with tf.name_scope('placeholder'):
            self.x_dm= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.x_bd= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_bd')
            self.f_val= tf.placeholder(tf.float32, shape=[None, 1], name='f_val')
            self.u_bd= tf.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.a_bd= tf.placeholder(tf.float32, shape=[None, 1], name='a_bd')
            self.n_vec= tf.placeholder(tf.float32, shape=[None, self.dim], name='n_vec')
        #*********************************************************************
        name_a='net_a'; name_u='net_u'; name_v='net_v';
        self.a_val= self.net_a(self.x_dm, 1, name_a, tf.AUTO_REUSE) 
        self.u_val, grad_u= self.grad_u(self.x_dm, name_u)
        #
        self.v_val_u, grad_v_u= self.grad_v(self.x_dm, name_v)
        w_val, grad_w= self.fun_w(self.x_dm, self.low, self.up)
        #
        u_bd_pred, grad_u_bd= self.grad_u(self.x_bd, name_u)
        a_bd_pred= self.net_a(self.x_bd, 1, name_a, tf.AUTO_REUSE)
        #**********************************************************************
        wv_val_u= tf.multiply(w_val, self.v_val_u)
        #
        dudw_val= tf.reduce_sum(tf.multiply(grad_u, grad_w), axis=1)
        dudw_val= tf.reshape(dudw_val, [-1,1])
        #
        dudv_val= tf.reduce_sum(tf.multiply(grad_u, grad_v_u), axis=1)
        dudv_val= tf.reshape(dudv_val, [-1,1])
        #
        dudwv_val= tf.add(tf.multiply(self.v_val_u, dudw_val),
                          tf.multiply(w_val, dudv_val))
        #
        _, _, g_obv= self.fun_g(self.x_bd, self.n_vec)
        g_val= tf.reduce_sum(tf.multiply(grad_u_bd, self.n_vec), axis=1)
        g_val= tf.reshape(g_val, [-1,1]) 
        #**********************************************************************
        with tf.variable_scope('loss'):
            with tf.name_scope('loss_u'):
                test_norm_u = tf.reduce_mean(wv_val_u**2)  # w*v_u 的l_2范数(v_u表示关于u的test function)
                #******************************************************************
                # operator-norm (a(x)固定，学习u(x))
                int_r1= tf.reduce_mean(tf.multiply(self.a_val, dudwv_val))
                int_l1= tf.reduce_mean(tf.multiply(self.f_val, wv_val_u))
                self.loss_int= self.beta_u*tf.square(int_l1-int_r1) / test_norm_u
                #*******************************************************************
                self.loss_u_bd= tf.reduce_mean(tf.abs(u_bd_pred-self.u_bd))  # loss on boundary for u(x)
                self.loss_g_bd= tf.reduce_mean(tf.abs(g_val - g_obv))
                #
                self.loss_a_bd= tf.reduce_mean(tf.abs(a_bd_pred-self.a_bd))  # loss on boundary for a(x)
                #
                self.loss_u= (self.beta_bd)*(self.loss_u_bd+self.loss_g_bd+self.loss_a_bd)+self.loss_int
            with tf.name_scope('loss_v'):
                # 
                self.loss_v_u=  - tf.log(self.loss_int)                      # loss for v_u
        #**************************************************************
        # 
        u_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_u)
        v_vars_u= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_v)
        a_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_a)
        #***************************************************************
        # 
        with tf.name_scope('optimizer'):
            self.ua_opt= tf.train.AdamOptimizer(self.u_rate).minimize(
                    self.loss_u, var_list= u_vars+a_vars)
            self.v_opt_u= tf.train.AdagradOptimizer(self.v_rate_u).minimize(
                    self.loss_v_u, var_list= v_vars_u)
    
    def train(self):
        #*********************************************************************
        tf.reset_default_graph(); self.build()
        #*********************************************************************
        # generate points for testing usage
        test_dict= self.sample_test(self.mesh_size, self.dim)
        #
        #saver= tf.train.Saver()
        list_dict={}; step_list=[]; 
        error_u=[]; error_a=[]
        loss_train=[]; loss_train_int=[]
        sample_time=[]; train_time=[]
        visual=visual_(self.dir)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iteration):
                #*************************************************************
                sample_time0= time.time()
                train_data= self.sample_train(self.dm_size, self.bd_size, self.dim)
                feed_train= {self.x_dm: train_data['x_dm'],
                            self.x_bd: train_data['x_bd'],
                            self.f_val: train_data['f_dm'],
                            self.u_bd: train_data['u_bd'],
                            self.a_bd: train_data['a_bd'],
                            self.n_vec: train_data['n_vector']}
                sample_time.append(time.time()-sample_time0)
                if i%5==0:
                    #
                    pred_u, pred_a= sess.run([self.u_val, self.a_val],
                                             feed_dict={self.x_dm: test_dict['test_x']})                 
                    err_u= np.sqrt(np.mean(np.square(test_dict['test_u']-pred_u)))
                    total_u= np.sqrt(np.mean(np.square(test_dict['test_u'])))
                    err_a= np.sqrt(np.mean(np.square(test_dict['test_a']-pred_a)))
                    total_a= np.sqrt(np.mean(np.square(test_dict['test_a'])))
                    step_list.append(i+1)
                    error_u.append(err_u/total_u)
                    error_a.append(err_a/total_a)
                    #************************************************
                    loss_u, loss_int, loss_a_bd= sess.run(
                        [self.loss_u, self.loss_int, self.loss_a_bd], 
                        feed_dict= feed_train)
                    loss_train.append(loss_u)
                    loss_train_int.append(loss_int)
                if i%500==0:
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} loss_int:{} loss_a_bd:{} l2r_a:{} l2r_u:{}'.format(
                        loss_u, loss_int, loss_a_bd, error_a[-1], error_u[-1]))
                    #
                    pred_u_draw, pred_a_draw, pred_v_draw= sess.run(
                            [self.u_val, self.a_val, self.v_val_u], 
                            feed_dict={self.x_dm: test_dict['draw_x']})
                    #visual.show_error(step_list, error_u, 'Relative error', 'l2r_u', self.dim)
                    #visual.show_error(step_list, error_a, 'Relative error', 'l2r_a', self.dim)
                    #visual.show_error(step_list, loss_train, 'Loss', 'loss', self.dim)
                #
                iter_time0= time.time()
                for _ in range(self.v_step_u):
                    _ = sess.run(self.v_opt_u, feed_dict=feed_train)                    
                for _ in range(self.u_step):
                    _ = sess.run(self.ua_opt, feed_dict=feed_train)
                train_time.append(time.time()-iter_time0)
                #
            #*******************************************
            #visual.show_error_abs(test_dict['mesh'], test_dict['draw_x'],
            #                      np.abs(test_dict['draw_a']-pred_a_draw), 'a', self.dim)
            #visual.show_error_abs(test_dict['mesh'], test_dict['draw_x'], 
            #                      np.abs(test_dict['draw_u']-pred_u_draw), 'u', self.dim)
            print('L2r_a is {}, L2r_u is {}'.format(np.min(error_a), np.min(error_u)))
            list_dict['error_u']= error_u
            list_dict['error_a']= error_a
            list_dict['loss_train']= loss_train
            list_dict['loss_train_int']= loss_train_int
            list_dict['step_list']= step_list
            list_dict['sample_time']= sample_time
            list_dict['train_time']= train_time
        return(test_dict, pred_u_draw, pred_a_draw, list_dict)

if __name__=='__main__':
    dim, noise, dm_size, bd_size= 5, 0, 100000, 50
    beta_u, beta_bd= 10, 10000
    u_step, u_rate, v_step, v_rate= 1, 0.001, 1, 0.008
    file_path='./iwan_piecewise/' # The filepath for saving data (and figures)
                                # create a file with title "iwan_smooth" before runing code
    iteration= 20001
    #
    demo= wan_inv(dim, noise/100, dm_size, bd_size, beta_u, beta_bd,
                  u_step, u_rate, v_step, v_rate, file_path, iteration)
    test_dict, pred_u_draw, pred_a_draw, list_dict= demo.train()
    #***************************
    # save data as .mat form
    import scipy.io
    data_save= {}
    data_save['test_dict']= test_dict
    data_save['pred_u_draw']= pred_u_draw
    data_save['pred_a_draw']= pred_a_draw
    data_save['list_dict']= list_dict
    scipy.io.savemat(file_path+'iwan_%dd_n%d'%(dim, noise), data_save)




