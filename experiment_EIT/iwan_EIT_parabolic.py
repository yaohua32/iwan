# -*- coding: utf-8 -*-
class visual_():

    def __init__(self, file_path):
        self.dir= file_path
        
    def show_error(self, iteration, error, dim, name):
        # 画 L_2 relative error vs. iteration 图像的函数
        # for drawing L_2 relative error vs. iteration
        plt.figure()
        plt.semilogy(iteration, error, color='b')
        plt.xlabel("Iteration", size=20)
        plt.ylabel("Relative error", size=20)        
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_iter_%s_%dd.png'%(name, dim))
        plt.close()
        
    def show_error_abs(self, mesh, x_y, z, name, dim):
        # 画pointwise absolute error 图像的函数
        # for drawing point-wise absolute error
        x= np.ravel(x_y[:,-1])
        y= np.ravel(x_y[:,0])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('t', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel(r'$x_1$', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_err/error_abs_%s_%dd.png'%(name, dim))
        plt.close()

    def show_u_val(self, mesh, z1, z2, name, i):
        # 画u(x,t)的函数
        x1, t = mesh
        z1= np.reshape(z1, [self.mesh_size, self.mesh_size])
        z2= np.reshape(z2, [self.mesh_size, self.mesh_size])
        #*******************
        fig= plt.figure(figsize=(12,5))
        ax1= fig.add_subplot(1,2,1)
        graph1= ax1.contourf(t, x1, z1, 10,  cmap= cm.jet)
        fig.colorbar(graph1, ax= ax1)
        #
        ax2= fig.add_subplot(1,2,2)
        graph2= ax2.contourf(t, x1, z2, 10,  cmap= cm.jet)
        fig.colorbar(graph2, ax= ax2)
        #*******************
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/%s_val_%d.png'%(name, name, i))
        plt.close()
        
    def show_v_val(self, mesh, x_y, z, name, i):
        # 画v(x)的函数
        x= np.ravel(x_y[:,-1])
        y= np.ravel(x_y[:,0])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('t', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel(r'$x_1$', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.dir+'figure_%s/%s_%d.png'%(name, name, i))
        plt.close()
        
        
class wan_inv():
    
    def __init__(self, file_name, dim, beta_u, beta_a, N_dm, N_bd, noise_level=0):
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
        from functools import reduce
        global reduce
        #
        self.dim= dim                            #问题的维度
        self.noise_level= noise_level
        self.low, self.up= 0.0, 1.0            #矩形区域[0, 1]^d
        self.t0, self.t1= 0.0, 1.0             #时间域为[0, 1]
        #
        self.scale= np.pi
        self.lam1, self.lam2= 0.2, 1.5      # lam1*exp(-lam2*t)
        self.k1, self.k2= 1.5, 0.6             #k(u)= k1+k2*u
        self.c1, self.c2= self.k1*(self.scale**2), self.k2*(self.scale**2)
                                               #C(u)= c1+c2*u
        #
        self.mesh_size= 50                      #用来生成testing data
        self.beta_u= beta_u                      #loss function for boundary of u(x) 前面的参数
        self.beta_a= beta_a                      #loss function for boundary of a(x) 前面的参数
        #
        self.outer_step_u= 1                     #外循环（解u(x)）
        self.outer_step_a= 1                     #外循环（解a(x)）
        self.v_layer= 6                          #test function v  的hidden layers 层数
        self.v_h_size= 20                        #test function v  每层的neuron 数目
        #
        self.a_layer= 6                         
        self.a_h_size= 20
        self.a_step= 2                           #解a(x)内循环（神经网络a的迭代步数）
        self.a_rate= 0.01                       #解a(x)内循环（神经网络a的learning rate）
        self.v_step_a= 1                         #解a(x)内循环（test function v的迭代步数）
        self.v_rate_a= 0.008                     #解a(x)内循环（test function v的leraning rate）
        #
        self.u_layer= 6                          
        self.u_h_size= 20                        
        self.u_step= 2                           #解u(x)内循环（神经网络u的迭代步数）
        self.u_rate= 0.01                       #解u(x)内循环（神经网络u的learning rate）
        self.v_step_u= 1                         #解u(x)内循环（test function v的迭代步数）
        self.v_rate_u= 0.008                     #解u(x)内循环（test function v的learning rate）
        #
        self.dm_size= N_dm                       #内部采样点数目                   
        self.bd_size= 2*dim*N_bd                 #边界采样点数目
        self.iteration= 20001
        #
        self.dir= file_name              #运行的时候需要建一个文件夹，以此名字命名，然后在该文件夹下面
                                                 #新建文件夹figure_err, figure_u, figure_a, figure_v，分别用来保存中间过程输出的图像
        
    def sample_train(self, dm_size, bd_size, dim):
        # 生成训练数据
        low, up, t0, t1= self.low, self.up, self.t0, self.t1
        #********************************************************
        # collocation points in domain
        x_dm= np.random.uniform(low, up, [dm_size, dim])
        t_dm= np.random.uniform(t0, t1,  [dm_size, 1])
        xt_dm= np.concatenate((x_dm, t_dm), axis=1)
        #*********************************************************
        # The value of f(x)
        term_1= self.lam1*np.exp(-self.lam2*t_dm)
        term_3= np.sin(self.scale*x_dm)
        u_val= np.multiply(term_1, np.reshape(np.sum(term_3, axis=1),[-1,1]))
        #
        term_1= np.multiply(self.k2, np.power(term_1, 2))
        term_3= 1.0-np.power(term_3, 2)
        term_2= (self.scale**2)*np.reshape(np.sum(term_3, axis=1),[-1,1])
        f_val= (self.c2*np.power(u_val,2)+self.c1*u_val
                -self.lam2*u_val-np.multiply(term_1, term_2))
        #***************************************************
        # initial condition
        x_init= np.random.uniform(low, up, [bd_size, dim])
        t_init= t0*np.ones([bd_size, 1])
        xt_init= np.concatenate((x_init, t_init), axis=1)
        term_1= self.lam1*np.exp(-self.lam2*t_init)
        term_3= np.sin(self.scale*x_init)
        u_init= np.multiply(term_1, np.reshape(np.sum(term_3, axis=1),[-1,1]))
        # end
        t_end= t1*np.ones([bd_size, 1])
        xt_end= np.concatenate((x_init, t_end), axis=1)
        #*********************************************************
        # collocation points on boundary
        xt_bd_list=[]
        n_vector_list=[]
        for i in range(dim):
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            t_bound= np.random.uniform(t0, t1, [bd_size, 1])
            x_bound[:,i]= up
            xt_bd_list.append(np.concatenate((x_bound, t_bound), axis=1))
            n_vector= np.zeros_like(x_bound)
            n_vector[:,i]=1
            n_vector_list.append(n_vector)
            x_bound= np.random.uniform(low, up, [bd_size, dim])
            t_bound= np.random.uniform(t0, t1, [bd_size, 1])
            x_bound[:,i]= low
            xt_bd_list.append(np.concatenate((x_bound, t_bound), axis=1))
            n_vector= np.zeros_like(x_bound)
            n_vector[:,i]=-1
            n_vector_list.append(n_vector)
        xt_bd= np.concatenate(xt_bd_list, axis=0)
        n_vector= np.concatenate(n_vector_list, 0)
        #*************************************************************
        # boundary condition of u(x) and k(u)
        term_1= self.lam1*np.exp(-self.lam2*xt_bd[:,[-1]])
        term_3= np.sin(self.scale*xt_bd[:,[i for i in range(dim)]])
        term_2= np.reshape(np.sum(term_3, axis=1),[-1,1])
        u_bd= np.reshape(np.multiply(term_1, term_2), [-1,1])
        #*********************************************************
        # observation of k(u) on boundary
        k_bd= self.k1+self.k2*u_bd
        #*********************************************************
        int_t= (t1-t0)
        int_x= (up-low)**dim
        #
        xt_dm= np.float32(xt_dm)
        xt_init= np.float32(xt_init)
        xt_end= np.float32(xt_end)
        xt_bd= np.float32(xt_bd)
        int_t= np.float32(int_t)
        int_x= np.float32(int_x)
        f_val= np.float32(f_val)
        u_init= np.float32(u_init)
        k_bd= np.float32(k_bd)
        n_vector= np.float32(n_vector)
        return(xt_dm, xt_init, xt_end, xt_bd, int_t, int_x, f_val, u_init, k_bd,u_bd, n_vector)
        
    def sample_test(self, mesh_size, dim):
        # 生成测试数据
        low, up, t0, t1= self.low, self.up, self.t0, self.t1
        #**********************************************************
        # generate meshgrid in the domain
        x_mesh= np.linspace(low, up, mesh_size)
        t_mesh= np.linspace(t0, t1,  mesh_size)
        mesh= np.meshgrid(x_mesh, t_mesh)
        #
        x1_dm= np.reshape(mesh[0], [-1,1])
        t_dm= np.reshape(mesh[1], [-1,1])
        #
        x2_dm= np.random.uniform(low, up, [self.mesh_size*self.mesh_size, dim-1])
        xt_dm= np.concatenate([x1_dm, x2_dm, t_dm], axis=1)
        x3_dm= 0.5*np.ones([self.mesh_size*self.mesh_size, dim-1])
        xt_draw_dm= np.concatenate([x1_dm, x3_dm, t_dm], axis=1)
        #***********************************************************
        # The exact u(x) and k(u)
        term_1= self.lam1*np.exp(-self.lam2*xt_dm[:,[-1]])
        term_3= np.sin(self.scale*xt_dm[:,[i for i in range(dim)]])
        term_2= np.reshape(np.sum(term_3, axis=1),[-1,1])
        u_dm= np.reshape(np.multiply(term_1, term_2), [-1,1])
        #
        term_1= self.lam1*np.exp(-self.lam2*xt_draw_dm[:,[-1]])
        term_3= np.sin(self.scale*xt_draw_dm[:,[i for i in range(dim)]])
        term_2= np.reshape(np.sum(term_3, axis=1),[-1,1])
        u_draw_dm= np.reshape(np.multiply(term_1, term_2), [-1,1])
        #
        k_dm= self.k1+self.k2*u_dm
        k_draw_dm= self.k1+self.k2*u_draw_dm
        #***********************************************************
        xt_dm= np.float32(xt_dm)
        xt_draw_dm= np.float32(xt_draw_dm)
        u_dm= np.float32(u_dm)
        u_draw_dm= np.float32(u_draw_dm)
        k_dm= np.float32(k_dm)
        k_draw_dm= np.float32(k_draw_dm)
        return(mesh, xt_dm, u_dm, k_dm, xt_draw_dm, u_draw_dm, k_draw_dm)
        
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
                    hi= tf.layers.dense(hi, h_size, activation= tf.nn.softplus, name='h_layer'+str(i))
                else:
                    hi= tf.sin(tf.layers.dense(hi, h_size), name='h_layer'+str(i))
            out= tf.layers.dense(hi, out_size, name='output_layer')
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
    
    def grad_u(self, xt_list, name, out_size=1):
        xt= tf.concat(xt_list, axis=1)
        #**************************************
        # u(x,t)
        u_val= self.net_u(xt, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_u(x,t)
        grad_u= tf.gradients(u_val, xt_list, unconnected_gradients='zero')
        du_x, du_t = grad_u[0], grad_u[-1]
        return(u_val, du_x, du_t)
        
    def grad_v(self, xt_list, name, out_size=1):        
        xt= tf.concat(xt_list, axis=1)
        #**************************************
        # v(x,t)
        v_val= self.net_v(xt, out_size, name, tf.AUTO_REUSE)
        #*************************************
        # grad_v(x,t)
        grad_v= tf.gradients(v_val, xt_list, unconnected_gradients='zero')
        dv_x, dv_t = grad_v[0], grad_v[-1]
        return(v_val, dv_x, dv_t)

    def fun_g(self, xt_list, n_vec):
        #x_list= tf.split(xt_list[0], self.dim, axis=1)
        #**************************************
        term_1= self.lam1*tf.exp(-self.lam2*xt_list[-1])
        term_3= tf.reduce_sum(tf.sin(self.scale*xt_list[0]), axis=1)
        u_val= tf.multiply(term_1, tf.reshape(term_3, [-1,1]))
        #
        du= tf.gradients(u_val, xt_list[0], unconnected_gradients='zero')[0]
        g_obv= tf.reduce_sum(tf.multiply(du, n_vec), axis=1)
        g_obv= tf.reshape(g_obv, [-1,1])
        return(u_val, du, g_obv)
    
    def build(self):
        #*********************************************************************
        with tf.name_scope('placeholder'):
            self.xt_dm= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='xt_dm')
            self.xt_init= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='xt_init')
            self.xt_end= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='xt_end')
            self.xt_bd= tf.placeholder(tf.float32, shape=[None, self.dim+1], name='xt_bd')
            self.int_t= tf.placeholder(tf.float32, shape=(), name='int_t')
            self.int_x= tf.placeholder(tf.float32, shape=(), name='int_x')
            self.f_val= tf.placeholder(tf.float32, shape=[None, 1], name='f_val')
            self.h_val= tf.placeholder(tf.float32, shape=[None, 1], name='h_val')
            self.k_bd= tf.placeholder(tf.float32, shape=[None, 1], name='k_bd')
            self.u_bd= tf.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.n_vec= tf.placeholder(tf.float32, shape=[None, self.dim], name='n_vec')
        #*********************************************************************
        name_a='net_a'; name_u='net_u'; name_vu='net_vu';  name_va='net_va'
        ################################################ initial condition
        xt_init_list= tf.split(self.xt_init, [self.dim, 1], axis=1)
        u_init= self.net_u(self.xt_init, 1, name_u, tf.AUTO_REUSE)
        w_init, _= self.fun_w(xt_init_list[0], self.low, self.up)
        v1_init= self.net_v(self.xt_init, 1, name_vu, tf.AUTO_REUSE)
        uwv1_init= tf.multiply(self.h_val, tf.multiply(w_init, v1_init))
        v2_init= self.net_v(self.xt_init, 1, name_va, tf.AUTO_REUSE)
        uwv2_init= tf.multiply(self.h_val, tf.multiply(w_init, v2_init))
        # end time
        xt_end_list= tf.split(self.xt_end, [self.dim, 1], axis=1)
        u_end= self.net_u(self.xt_end, 1, name_u, tf.AUTO_REUSE)
        u_end_true= u_end
        #u_end_true,_,_= self.fun_g(xt_end_list, self.n_vec)##########################(for debugging)
        w_end, _= self.fun_w(xt_end_list[0], self.low, self.up)
        v1_end= self.net_v(self.xt_end, 1, name_vu, tf.AUTO_REUSE)
        uwv1_end= tf.multiply(u_end, tf.multiply(w_end, v1_end))
        v2_end= self.net_v(self.xt_end, 1, name_va, tf.AUTO_REUSE)
        uwv2_end= tf.multiply(u_end_true, tf.multiply(w_end, v2_end))
        ################################################# boundary condition
        xt_bd_list= tf.split(self.xt_bd, [self.dim, 1], axis=1)
        #bd cond for k(u)
        k_bd_pred= self.net_a(self.xt_bd, 1, name_a, tf.AUTO_REUSE)        
        #bd cond for u(x,t)
        _, _, g_obv= self.fun_g(xt_bd_list, self.n_vec)
        u_bd_pred, du_bd, _= self.grad_u(xt_bd_list, name_u)
        g_pred= tf.reduce_sum(tf.multiply(du_bd, self.n_vec), axis=1)
        g_pred= tf.reshape(g_pred, [-1,1])        
        ################################################# inside domain
        xt_dm_list= tf.split(self.xt_dm, [self.dim, 1], axis=1)
        self.w_dm, dwx_dm= self.fun_w(xt_dm_list[0], self.low, self.up)
        self.v1_dm, dv1x_dm, dv1t_dm= self.grad_v(xt_dm_list, name_vu)
        self.v2_dm, dv2x_dm, dv2t_dm= self.grad_v(xt_dm_list, name_va)
        self.u_dm, dux_dm, dut_dm= self.grad_u(xt_dm_list, name_u)
        u_dm_true, dux_dm_true= self.u_dm, dux_dm
        #u_dm_true, dux_dm_true,_= self.fun_g(xt_dm_list, self.n_vec)##########################(for debugging)
        self.wv1_dm= tf.multiply(self.w_dm, self.v1_dm)
        self.wv2_dm= tf.multiply(self.w_dm, self.v2_dm)
        #
        self.k_val= self.net_a(self.xt_dm, 1, name_a, tf.AUTO_REUSE) 
        dux_dw_dm= tf.reduce_sum(tf.multiply(dux_dm, dwx_dm), axis=1)
        dux_dw_dm= tf.reshape(dux_dw_dm, [-1,1])
        dux_dv1_dm= tf.reduce_sum(tf.multiply(dux_dm, dv1x_dm), axis=1)
        dux_dv1_dm= tf.reshape(dux_dv1_dm, [-1,1])
        dux_dwv1_dm= tf.add(tf.multiply(self.v1_dm, dux_dw_dm),
                          tf.multiply(self.w_dm, dux_dv1_dm))
        dux_dv2_dm= tf.reduce_sum(tf.multiply(dux_dm_true, dv2x_dm), axis=1)
        dux_dv2_dm= tf.reshape(dux_dv2_dm, [-1,1])
        dux_dwv2_dm= tf.add(tf.multiply(self.v2_dm, dux_dw_dm),
                          tf.multiply(self.w_dm, dux_dv2_dm))
        #
        k_dudwv1_dm= tf.multiply(self.k_val, dux_dwv1_dm)
        k_dudwv2_dm= tf.multiply(self.k_val, dux_dwv2_dm)
        #
        uw_dv1t_dm= tf.multiply(tf.multiply(self.u_dm, self.w_dm), dv1t_dm)
        uw_dv2t_dm= tf.multiply(tf.multiply(u_dm_true, self.w_dm), dv2t_dm)
        #
        f_wv1_dm= tf.multiply(self.f_val, self.wv1_dm)
        f_wv2_dm= tf.multiply(self.f_val, self.wv2_dm)
        #####################################################
        int_dm= tf.multiply(self.int_x, self.int_t)
        #**********************************************************************
        with tf.variable_scope('loss'):
            with tf.name_scope('loss_u'):
                test_norm_u = tf.multiply(tf.reduce_mean(self.wv1_dm**2), int_dm)  
                test_norm_k = tf.multiply(tf.reduce_mean(self.wv2_dm**2), int_dm)
                #******************************************************************
                # operator-norm (a(x)固定，学习u(x))
                int_l1= tf.multiply(tf.reduce_mean(uwv1_end), self.int_x)
                int_l2= tf.multiply(tf.reduce_mean(k_dudwv1_dm), int_dm)
                int_r1= tf.multiply(tf.reduce_mean(uwv1_init), self.int_x)
                int_r2= tf.multiply(tf.reduce_mean(uw_dv1t_dm), int_dm)
                int_r3= tf.multiply(tf.reduce_mean(f_wv1_dm), int_dm)  
                #
                self.loss_int= 1.0*tf.square(int_l1+int_l2-int_r1-int_r2-int_r3) / test_norm_u
                #*******************************************************************
                # operator-norm (u(x)固定，学习a(x))
                int_l1_k= tf.multiply(tf.reduce_mean(uwv2_end), self.int_x)
                int_l2_k= tf.multiply(tf.reduce_mean(k_dudwv2_dm), int_dm)
                int_r1_k= tf.multiply(tf.reduce_mean(uwv2_init), self.int_x)
                int_r2_k= tf.multiply(tf.reduce_mean(uw_dv2t_dm), int_dm)
                int_r3_k= tf.multiply(tf.reduce_mean(f_wv2_dm), int_dm)   
                #
                self.loss_intk= 1.0*tf.square(int_l1_k+int_l2_k-int_r1_k-int_r2_k-int_r3_k) / test_norm_k
                #*********************************************************************
                #
                self.loss_u_bd= tf.reduce_mean(tf.abs(u_bd_pred-self.u_bd))
                self.loss_g_bd= tf.reduce_mean(tf.abs(g_pred - g_obv))
                #
                self.loss_k_bd= tf.reduce_mean(tf.abs(k_bd_pred-self.k_bd))
                #
                self.loss_u_init= tf.reduce_mean(tf.abs(u_init-self.h_val))
                #
                self.loss_u= (10000)*(self.loss_u_init+self.loss_g_bd+self.loss_u_bd)+self.loss_int
                #
                self.loss_k= (1000)*self.loss_k_bd+self.loss_intk
            with tf.name_scope('loss_v'):
                # 
                self.loss_v_u=  - tf.log(self.loss_int)                      # loss for v_u
                self.loss_v_k=  - tf.log(self.loss_intk)                     # loss for v_a
        #**************************************************************
        # 
        u_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_u)
        v_vars_u= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_vu)
        a_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_a)
        v_vars_a= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_va)
        #***************************************************************
        # 
        with tf.name_scope('optimizer'):
            self.u_opt= tf.train.AdagradOptimizer(self.u_rate).minimize(
                    self.loss_u, var_list= u_vars)
            self.v_opt_u= tf.train.AdagradOptimizer(self.v_rate_u).minimize(
                    self.loss_v_u, var_list= v_vars_u)
            self.a_opt= tf.train.AdagradOptimizer(self.a_rate).minimize(
                    self.loss_k, var_list= a_vars)
            self.v_opt_a= tf.train.AdagradOptimizer(self.v_rate_a).minimize(
                    self.loss_v_k, var_list= v_vars_a)
    
    def train(self):
        #*********************************************************************
        tf.reset_default_graph(); self.build()
        #*********************************************************************
        # generate points for testing usage
        mesh, test_xt, test_u, test_k, draw_xt, draw_u, draw_k= self.sample_test(self.mesh_size, self.dim)
        #saver= tf.train.Saver()
        step=[]; error_u=[]; error_k=[]
        time_begin=time.time(); time_list=[]; iter_time_list=[]
        visual=visual_(self.dir)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iteration):
                train_data= self.sample_train(self.dm_size, self.bd_size, self.dim)
                feed_train={self.xt_dm: train_data[0],
                            self.xt_init: train_data[1],
                            self.xt_end: train_data[2],
                            self.xt_bd: train_data[3],
                            self.int_t: train_data[4],
                            self.int_x: train_data[5],
                            self.f_val: train_data[6],
                            self.h_val: train_data[7],
                            self.k_bd: train_data[8],
                            self.u_bd: train_data[9],
                            self.n_vec: train_data[10]}              
                if i%5==0:
                    #
                    pred_u, pred_k= sess.run([self.u_dm, self.k_val],feed_dict={self.xt_dm: test_xt})                 
                    err_u= np.sqrt(np.mean(np.square(test_u-pred_u)))
                    total_u= np.sqrt(np.mean(np.square(test_u)))
                    err_k= np.sqrt(np.mean(np.square(test_k-pred_k)))
                    total_k= np.sqrt(np.mean(np.square(test_k)))
                    step.append(i+1)
                    error_u.append(err_u/total_u)
                    error_k.append(err_k/total_k)
                    time_step= time.time(); time_list.append(time_step-time_begin)
                if i%500==0:
                    loss_u, loss_v, loss_k, loss_k_bd= sess.run(
                        [self.loss_u, self.loss_v_k, self.loss_k, self.loss_k_bd], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} v_loss:{} k_loss:{} loss_k_bd:{} l2r_k:{} l2r_u:{}'.format(
                        loss_u, loss_v, loss_k, loss_k_bd, error_k[-1], error_u[-1]))
                    #
                    pred_u_draw, pred_k_draw= sess.run(
                            [self.u_dm, self.k_val], 
                            feed_dict={self.xt_dm: draw_xt})
                    #visual.show_error(step, error_u, self.dim, 'l2r_u')
                    #visual.show_error(step, error_k, self.dim, 'l2r_k')
                    #visual.show_u_val(mesh, draw_k, pred_k_draw, 'k',  i)
                    #visual.show_u_val(mesh, draw_u, pred_u_draw, 'u',  i)
                #
                iter_time0= time.time()
                for _ in range(self.outer_step_u):
                    for _ in range(self.v_step_u):
                        _ = sess.run(self.v_opt_u, feed_dict=feed_train)                    
                    for _ in range(self.u_step):
                        _ = sess.run(self.u_opt, feed_dict=feed_train)
                for _ in range(self.outer_step_a):
                    for _ in range(self.v_step_a):
                        _ = sess.run(self.v_opt_a, feed_dict=feed_train)                    
                    for _ in range(self.a_step):
                        _ = sess.run(self.a_opt, feed_dict=feed_train)
                iter_time_list.append(time.time()-iter_time0)
                #
            #*******************************************
            #visual.show_error_abs(mesh, draw_xt, np.abs(draw_k-pred_k_draw), 'k', self.dim)
            #visual.show_error_abs(mesh, draw_xt, np.abs(draw_u-pred_u_draw), 'u', self.dim)
            print('L2r_k is {}, L2r_u is {}'.format(np.min(error_k), np.min(error_u)))
        return(mesh, test_xt, draw_xt, test_u, draw_u, test_k, draw_k, pred_u, pred_u_draw, pred_k, pred_k_draw, 
               step, error_k, error_u, time_list, iter_time_list)

if __name__=='__main__':
    dim, beta_u, beta_f, N_dm, N_bd= 5, 10000, 10000, 100000, 50
    file_name= './problem_EIT_parabolic/'
    demo= wan_inv(file_name, dim, beta_u, beta_f, N_dm, N_bd)
    mesh, test_xt, draw_xt, test_u, draw_u, test_k, draw_k, pred_u, pred_u_draw, pred_k, pred_k_draw, step, error_k, error_u, time_list, iter_time_list= demo.train()
    #***************************
    # save data as .mat form
    import scipy.io
    data_save= {}
    data_save['mesh']= mesh
    data_save['test_xt']= test_xt
    data_save['test_u']= test_u
    data_save['test_k']= test_k
    data_save['pred_u']= pred_u
    data_save['pred_k']= pred_k
    data_save['draw_xt']= draw_xt
    data_save['draw_u']= draw_u
    data_save['draw_k']= draw_k
    data_save['pred_u_draw']= pred_u_draw
    data_save['pred_k_draw']= pred_k_draw
    data_save['step']= step
    data_save['error_k']= error_k
    data_save['error_u']= error_u
    data_save['time_list']= time_list
    data_save['iter_time_list']= iter_time_list
    scipy.io.savemat(file_name+'iwan_%dd'%(dim), data_save)




