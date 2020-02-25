# -*- coding: utf-8 -*-
class visual_():
    
    def __init__(self, file_path):
        self.dir= file_path
        
    def show_error(self, iteration, error, dim, name):
        # 画 L_2 relative error vs. iteration 图像的函数
        # This function designed for drawing L_2 relative error vs. iteration
        plt.figure()
        plt.semilogy(iteration, error, color='b')
        plt.xlabel("Iteration", size=20)
        plt.ylabel("Relative error", size=20)        
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_iter_%s_%dd.png'%(name, dim))
        plt.close()
        
    def show_error_abs(self, mesh, x_y, z, name, dim):
        # 画pointwise absolute error 图像的函数
        # This function designed for drawing point-wise absolute error
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_abs_%s_%dd.png'%(name, dim))
        plt.close()


    def show_u_val(self, mesh, z1, z2, name, i):
        # 画u(x)的函数
        x1, x2 = mesh
        z1= np.reshape(z1, [self.mesh_size, self.mesh_size])
        z2= np.reshape(z2, [self.mesh_size, self.mesh_size])
        #*******************
        fig= plt.figure(figsize=(12,5))
        ax1= fig.add_subplot(1,2,1)
        graph1= ax1.contourf(x1, x2, z1, 10,  cmap= cm.jet)
        fig.colorbar(graph1, ax= ax1)
        #
        ax2= fig.add_subplot(1,2,2)
        graph2= ax2.contourf(x1, x2, z2, 10,  cmap= cm.jet)
        fig.colorbar(graph2, ax= ax2)
        #*******************
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/%s_val_%d.png'%(name, name, i))
        plt.close()
        
    def show_v_val(self, mesh, x_y, z, name, i):
        # 画v(x)的函数
        # This function designed for drawing the figure of test function v(x)
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/%s_%d.png'%(name, name, i))
        plt.close()
    
class wan_inv():
    
    def __init__(self, file_name, dim, beta, N_dm, N_bd):
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
        self.dim= dim                           #问题的维度
        self.low, self.up=  0.0, 1.0           #矩形区域[-1,1]^d
        self.la= np.pi
        #
        self.mesh_size= 50                      #用来生成testing data
        self.beta= beta
        #
        self.v_layer= 6                          #test function v  的hidden layers 层数
        self.v_h_size= 20                        #test function v  每层的neuron 数目
        self.v_step= 1                      
        self.v_rate= 0.008     
        #
        self.a_layer= 6                         
        self.a_h_size= 20
        self.u_layer= 6                          
        self.u_h_size= 20      
        #        
        self.ua_step= 2                          
        self.ua_rate= 0.01                    
        #
        self.dm_size= N_dm                       #内部采样点数目                   
        self.bd_size= N_bd                       #边界采样点数目
        self.iteration= 20001
        #
        self.dir= file_name              #运行的时候需要建一个文件夹，以此名字命名，然后在该文件夹下面
                                                 #新建文件夹figure_err, figure_u, figure_a, figure_v，分别用来保存中间过程输出的图像
        
    def sample_train(self, dm_size, bd_size, dim):
        # 生成训练数据
        low, up= self.low, self.up
        #********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
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
        param= (dim-1)*self.la**2/2
        u_bd= np.exp(param*x_bd[:,0]*(x_bd[:,0]-1))
        for i in range(dim-1):
            u_bd= np.multiply(u_bd, np.sin(self.la*x_bd[:,i+1]))
        u_bd= np.reshape(u_bd, [-1, 1])
        #*********************************************************
        # observation of a(x) on boundary
        a_bd= np.exp(param*x_bd[:,0]*(1-x_bd[:,0]))
        a_bd= np.reshape(a_bd/self.la, [-1,1])
        #*********************************************************
        int_dm= (up-low)**dim
        #
        x_dm= np.float32(x_dm)
        x_bd= np.float32(x_bd)
        int_dm= np.float32(int_dm)
        u_bd= np.float32(u_bd)
        a_bd= np.float32(a_bd)
        n_vector= np.float32(n_vector)
        return(x_dm, x_bd, int_dm, u_bd, a_bd, n_vector)
        
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
        param= (dim-1)*self.la**2/2
        u_dm= np.exp(param*x_dm[:,0]*(x_dm[:,0]-1))
        u_draw_dm= np.exp(param*x_draw_dm[:,0]*(x_draw_dm[:,0]-1))
        for i in range(dim-1):
            u_dm= np.multiply(u_dm, np.sin(self.la*x_dm[:,i+1]))
            u_draw_dm= np.multiply(u_draw_dm, np.sin(self.la*x_draw_dm[:,i+1]))
        u_dm= np.reshape(u_dm, [-1, 1])
        u_draw_dm= np.reshape(u_draw_dm, [-1, 1])
        #***********************************************************
        # The exact a(x)
        a_dm= np.exp(param*x_dm[:,0]*(1-x_dm[:,0]))
        a_dm= np.reshape(a_dm/self.la, [-1,1])  
        a_draw_dm= np.exp(param*x_draw_dm[:,0]*(1-x_draw_dm[:,0]))
        a_draw_dm= np.reshape(a_draw_dm/self.la, [-1,1])
        #***********************************************************
        x_dm= np.float32(x_dm)
        x_draw_dm= np.float32(x_draw_dm)
        u_dm= np.float32(u_dm)
        u_draw_dm= np.float32(u_draw_dm)
        a_dm= np.float32(a_dm)
        a_draw_dm= np.float32(a_draw_dm)
        return(mesh, x_dm, u_dm, a_dm, x_draw_dm, u_draw_dm, a_draw_dm)
 
    def net_a(self, x_in, out_size, name, reuse):
        # 逼近 a(x) 的神经网络
        #*****************************************************
        # Neural Net for a(x)
        h_size= self.a_h_size
        with tf.variable_scope(name, reuse=reuse):
            hi= tf.layers.dense(x_in, h_size, activation= tf.nn.tanh, name='input_layer')
            hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='input_layer1')
            for i in range(self.a_layer):
                if i%2==0:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.elu, name='h_layer'+str(i))
                else:
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.tanh, name='h_layer'+str(i))
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
        param= (self.dim-1)*self.la**2/2
        u_val= tf.exp(tf.multiply(param, tf.multiply(x_list[0], x_list[0]-1)))
        for i in range(self.dim-1):
            u_val= tf.multiply(u_val, tf.sin(tf.multiply(self.la, x_list[i+1])))
        u_val= tf.reshape(u_val, [-1,1])
        #
        du= tf.gradients(u_val, x, unconnected_gradients='zero')[0]
        du= tf.where(tf.is_nan(du), tf.zeros_like(du), du)
        g_obv= tf.reduce_sum(tf.multiply(du, n_vec), axis=1)
        g_obv= tf.reshape(g_obv, [-1,1])
        return(u_val, du, g_obv)
    
    def build(self):
        #*********************************************************************
        with tf.name_scope('placeholder'):
            self.x_dm= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.x_bd= tf.placeholder(tf.float32, shape=[None, self.dim], name='x_bd')
            self.int_dm= tf.placeholder(tf.float32, shape=(), name='int_dm')
            self.u_bd= tf.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.a_bd= tf.placeholder(tf.float32, shape=[None, 1], name='a_bd')
            self.n_vec= tf.placeholder(tf.float32, shape=[None, self.dim], name='n_vec')
        #*********************************************************************
        name_a='net_a'; name_u='net_u'; name_v='net_v';
        self.a_val= self.net_a(self.x_dm, 1, name_a, tf.AUTO_REUSE) 
        self.u_val, grad_u= self.grad_u(self.x_dm, name_u)
        self.v_val, grad_v= self.grad_v(self.x_dm, name_v)
        w_val, grad_w= self.fun_w(self.x_dm, self.low, self.up)
        u_bd_pred, grad_u_bd= self.grad_u(self.x_bd, name_u)
        a_bd_pred= self.net_a(self.x_bd, 1, name_a, tf.AUTO_REUSE)
        #**********************************************************************
        wv_val= tf.multiply(w_val, self.v_val)
        #
        dudw_val= tf.reduce_sum(tf.multiply(grad_u, grad_w), axis=1)
        dudw_val= tf.reshape(dudw_val, [-1,1])
        #
        dudv_val= tf.reduce_sum(tf.multiply(grad_u, grad_v), axis=1)
        dudv_val= tf.reshape(dudv_val, [-1,1])
        #
        dudwv_val= tf.add(tf.multiply(self.v_val, dudw_val),
                          tf.multiply(w_val, dudv_val))
        #
        _, _, g_obv= self.fun_g(self.x_bd, self.n_vec)
        g_val= tf.reduce_sum(tf.multiply(grad_u_bd, self.n_vec), axis=1)
        g_val= tf.reshape(g_val, [-1,1]) 
        #**********************************************************************
        with tf.variable_scope('loss'):
            with tf.name_scope('loss_u'):
                test_norm = tf.multiply(tf.reduce_mean(wv_val**2), self.int_dm)  # w*v_u 的l_2范数(v_u表示关于u的test function)
                #******************************************************************
                # operator-norm (a(x)固定，学习u(x))
                int_r1= tf.multiply(tf.reduce_mean(tf.multiply(self.a_val, dudwv_val)), self.int_dm)
                #int_l1= tf.multiply(tf.reduce_mean(tf.multiply(self.f_val, wv_val_u)), self.int_dm)
                self.loss_int= 10*tf.square(int_r1) / test_norm
                #*******************************************************************
                #
                self.loss_u_bd= tf.reduce_mean(tf.abs(u_bd_pred-self.u_bd))  # loss on boundary for u(x)
                self.loss_g_bd= tf.reduce_mean(tf.abs(g_val - g_obv))
                #
                self.loss_a_bd= tf.reduce_mean(tf.abs(a_bd_pred-self.a_bd))  # loss on boundary for a(x)
                #
                self.loss_total= (self.beta)*(self.loss_u_bd+self.loss_g_bd+self.loss_a_bd)+self.loss_int
            with tf.name_scope('loss_v'):
                # 
                self.loss_v=  - tf.log(self.loss_int)                      # loss for v
        #**************************************************************
        # 
        u_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_u)
        v_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_v)
        a_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_a)
        #***************************************************************
        # 
        with tf.name_scope('optimizer'):
            self.ua_opt= tf.train.AdagradOptimizer(self.ua_rate).minimize(
                    self.loss_total, var_list= u_vars+a_vars)
            self.v_opt= tf.train.AdagradOptimizer(self.v_rate).minimize(
                    self.loss_v, var_list= v_vars)
    
    def train(self):
        #*********************************************************************
        tf.reset_default_graph(); self.build()
        #*********************************************************************
        # generate points for testing usage
        mesh, test_x, test_u, test_a, draw_x, draw_u, draw_a= self.sample_test(self.mesh_size, self.dim)
        #
        #saver= tf.train.Saver()
        step=[]; error_u=[]; error_a=[]
        time_begin=time.time(); time_list=[]; iter_time_list=[]
        visual=visual_(self.dir)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iteration):
                train_data= self.sample_train(self.dm_size, self.bd_size, self.dim)
                feed_train= {self.x_dm: train_data[0],
                            self.x_bd: train_data[1],
                            self.int_dm: train_data[2],
                            self.u_bd: train_data[3],
                            self.a_bd: train_data[4],
                            self.n_vec: train_data[5]}
                if i%5==0:
                    #
                    pred_u, pred_a= sess.run([self.u_val, self.a_val],feed_dict={self.x_dm: test_x})                 
                    err_u= np.sqrt(np.mean(np.square(test_u-pred_u)))
                    total_u= np.sqrt(np.mean(np.square(test_u)))
                    err_a= np.sqrt(np.mean(np.square(test_a-pred_a)))
                    total_a= np.sqrt(np.mean(np.square(test_a)))
                    step.append(i+1)
                    error_u.append(err_u/total_u)
                    error_a.append(err_a/total_a)
                    time_step= time.time(); time_list.append(time_step-time_begin)
                if i%500==0:
                    loss_total, loss_v, loss_int, loss_a_bd= sess.run(
                        [self.loss_total, self.loss_v, self.loss_int, self.loss_a_bd], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('loss_total:{} loss_v:{} loss_int:{} loss_a_bd:{} l2r_a:{} l2r_u:{}'.format(
                        loss_total, loss_v, loss_int, loss_a_bd, error_a[-1], error_u[-1]))
                    #
                    pred_u_draw, pred_a_draw, pred_v_draw= sess.run(
                            [self.u_val, self.a_val, self.v_val], 
                            feed_dict={self.x_dm: draw_x})
                    #visual.show_error(step, error_u, self.dim, 'l2r_u')
                    #visual.show_error(step, error_a, self.dim, 'l2r_a')
                    #visual.show_u_val(mesh, draw_a, pred_a_draw, 'a',  i)
                    #visual.show_u_val(mesh, draw_u, pred_u_draw, 'u',  i)
                #
                iter_time0= time.time()
                for _ in range(self.v_step):
                    _ = sess.run(self.v_opt, feed_dict=feed_train)                    
                for _ in range(self.ua_step):
                    _ = sess.run(self.ua_opt, feed_dict=feed_train)
                iter_time_list.append(time.time()-iter_time0)
                #
            #*******************************************
            #visual.show_error_abs(mesh, draw_x, np.abs(draw_a-pred_a_draw), 'a', self.dim)
            #visual.show_error_abs(mesh, draw_x, np.abs(draw_u-pred_u_draw), 'u', self.dim)
            print('L2r_a is {}, L2r_u is {}'.format(np.min(error_a), np.min(error_u)))
        return(mesh, test_x, draw_x, test_u, draw_u, test_a, draw_a, pred_u, pred_u_draw, pred_a, pred_a_draw, 
               step, error_a, error_u, time_list, iter_time_list, self.dim)

if __name__=='__main__':
    dim, beta, N_dm, N_bd= 5, 10000, 100000, 50
    file_name= './problem_EIT/'
    demo= wan_inv(file_name, beta, N_dm, N_bd)
    mesh, test_x, draw_x, test_u, draw_u, test_a, draw_a, pred_u, pred_u_draw, pred_a, pred_a_draw, step, error_a, error_u, time_list, iter_time_list, dim= demo.train()
    #***************************
    # save data as .mat form
    import scipy.io
    data_save= {}
    data_save['mesh']= mesh
    data_save['test_x']= test_x
    data_save['test_u']= test_u
    data_save['test_a']= test_a
    data_save['pred_u']= pred_u
    data_save['pred_a']= pred_a
    data_save['draw_x']= draw_x
    data_save['draw_u']= draw_u
    data_save['draw_a']= draw_a
    data_save['pred_u_draw']= pred_u_draw
    data_save['pred_a_draw']= pred_a_draw
    data_save['step']= step
    data_save['error_a']= error_a
    data_save['error_u']= error_u
    data_save['time_list']= time_list
    data_save['iter_time_list']= iter_time_list
    scipy.io.savemat(file_name+'iwan_%dd'%(dim), data_save)
