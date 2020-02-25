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
        # for drawing the figure of test function v(x)
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
    
    def __init__(self, file_name, dim, beta_u, beta_a, N_dm, N_bd, noise_level):
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
        self.k1=   [2.0, 1.0]+[0.01]*(dim-2)     #\omega_1区域表达式中:前面的系数
        self.k2=   [1.0, 2.0]+[0.01]*(dim-2)
        self.c_a1= [0.5, 0.5]+[0.0]*(dim-2)       #\omega_1区域表达式中:区域的中心
        self.c_a2= [-0.5, -0.5]+[0.0]*(dim-2)
        self.c_u= [0, 0]+[0.0]*(dim-2)           #真实解表达式中:最小值点
        self.r1= 0.4                              #\omega_1区域表达式中：半径值
        self.r2= 0.4
        self.alpha= 0.05                          #用来控制不连续程度的值（越小奇异性越大）
        self.a1= 4.0                             #coefficient a(x) 在\omega_1区域内的值
        self.a2= 2.0
        self.a0= 0.5                             #coefficient a(x)  在\omega_1区域之外的值
        self.mesh_size= 100                      #用来生成testing data
        self.beta_u= beta_u                      #loss function for boundary of u(x) 前面的参数
        self.beta_a= beta_a                      #loss function for boundary of a(x) 前面的参数
        #
        self.outer_step_u= 1                     #外循环（解u(x)）
        self.outer_step_a= 1                     #外循环（解a(x)）
        self.v_layer= 6                          #test function v  的hidden layers 层数
        self.v_h_size= 20                        #test function v  每层的neuron 数目
        #
        self.a_layer= 4                         
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
        self.bd_size= N_bd                       #边界采样点数目
        self.iteration= 20001
        #
        self.dir= file_name              #运行的时候需要建一个文件夹，以此名字命名，然后在该文件夹下面
                                                 #新建文件夹figure_err, figure_u, figure_a, figure_v，分别用来保存中间过程输出的图像
        
    def get_truncated_normal(self, mean=0.0, sd=0.1):
        # 观测噪音生成函数
        # This function is designed for adding noise
        low=-100; up= 100
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
        omega_a1, omega_a2, omega_u1, omega_u2= 0.0, 0.0, 0.0, 0.0
        for i in range(dim):
            omega_a1= omega_a1+self.k1[i]**2*(x_dm[:,i]-self.c_a1[i])**2
            omega_a2= omega_a2+self.k2[i]**2*(x_dm[:,i]-self.c_a2[i])**2
            omega_u1= omega_u1+self.k1[i]**2*(x_dm[:,i]-self.c_a1[i])*(x_dm[:,i]-self.c_u[i])
            omega_u2= omega_u2+self.k2[i]**2*(x_dm[:,i]-self.c_a2[i])*(x_dm[:,i]-self.c_u[i])
        exp_term1= np.exp((omega_a1-self.r1**2)/self.alpha)
        exp_term2= np.exp((omega_a2-self.r2**2)/self.alpha)
        #
        part_one= (4*(self.a1-self.a0)*omega_u1/(self.alpha/exp_term1+2*self.alpha+self.alpha*exp_term1)+
                   4*(self.a1-self.a2)*omega_u2/(self.alpha/exp_term2+2*self.alpha+self.alpha*exp_term2))
        part_two= 2*dim*(self.a0*(1-1/(1+exp_term1)-1/(1+exp_term2))+self.a1/(1+exp_term1)+self.a2/(1+exp_term2))
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
        u_bd= u_bd*(1.0+self.noise_level*distb.rvs(np.shape(u_bd)))# adding noise
        u_bd= np.reshape(u_bd, [-1, 1])
        #*********************************************************
        # observation of a(x) on boundary
        omega_a1_bd, omega_a2_bd= 0.0, 0.0
        for i in range(dim):
            omega_a1_bd= omega_a1_bd+self.k1[i]**2*(x_bd[:,i]-self.c_a1[i])**2
            omega_a2_bd= omega_a2_bd+self.k2[i]**2*(x_bd[:,i]-self.c_a2[i])**2
        exp_term1_bd= np.exp((omega_a1_bd-self.r1**2)/self.alpha)
        exp_term2_bd= np.exp((omega_a2_bd-self.r2**2)/self.alpha)
        #
        a_bd= (self.a0*(1-1/(1+exp_term1_bd)-1/(1+exp_term2_bd))+
               self.a1/(1+exp_term1_bd)+self.a2/(1+exp_term2_bd))
        a_bd= a_bd*(1.0+ self.noise_level*distb.rvs(np.shape(a_bd)))# adding noise
        a_bd= np.reshape(a_bd, [-1,1])
        #*********************************************************
        int_dm= (up-low)**dim
        #
        x_dm= np.float32(x_dm)
        x_bd= np.float32(x_bd)
        int_dm= np.float32(int_dm)
        f_dm= np.float32(f_dm)
        u_bd= np.float32(u_bd)
        a_bd= np.float32(a_bd)
        n_vector= np.float32(n_vector)
        return(x_dm, x_bd, int_dm, f_dm, u_bd, a_bd, n_vector)
        
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
        omega_a1, omega_a2= 0.0, 0.0
        omega_draw_a1, omega_draw_a2= 0.0, 0.0
        for i in range(dim):
            omega_a1= omega_a1+self.k1[i]**2*(x_dm[:,i]-self.c_a1[i])**2
            omega_a2= omega_a2+self.k2[i]**2*(x_dm[:,i]-self.c_a2[i])**2
            omega_draw_a1= omega_draw_a1+self.k1[i]**2*(x_draw_dm[:,i]-self.c_a1[i])**2
            omega_draw_a2= omega_draw_a2+self.k2[i]**2*(x_draw_dm[:,i]-self.c_a2[i])**2
        exp_term1= np.exp((omega_a1-self.r1**2)/self.alpha)
        exp_term2= np.exp((omega_a2-self.r2**2)/self.alpha)
        exp_draw_term1= np.exp((omega_draw_a1-self.r1**2)/self.alpha)
        exp_draw_term2= np.exp((omega_draw_a2-self.r2**2)/self.alpha)
        #
        a_dm= (self.a0*(1-1/(1+exp_term1)-1/(1+exp_term2))+
               self.a1/(1+exp_term1)+self.a2/(1+exp_term2))
        a_dm= np.reshape(a_dm, [-1,1])
        a_draw_dm= (self.a0*(1-1/(1+exp_draw_term1)-1/(1+exp_draw_term2))+
                    self.a1/(1+exp_draw_term1)+self.a2/(1+exp_draw_term2))
        a_draw_dm= np.reshape(a_draw_dm, [-1,1])
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
            self.int_dm= tf.placeholder(tf.float32, shape=(), name='int_dm')
            self.f_val= tf.placeholder(tf.float32, shape=[None, 1], name='f_val')
            self.u_bd= tf.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.a_bd= tf.placeholder(tf.float32, shape=[None, 1], name='a_bd')
            self.n_vec= tf.placeholder(tf.float32, shape=[None, self.dim], name='n_vec')
        #*********************************************************************
        name_a='net_a'; name_u='net_u'; name_vu='net_vu';  name_va='net_va' 
        self.a_val= self.net_a(self.x_dm, 1, name_a, tf.AUTO_REUSE) 
        self.u_val, grad_u= self.grad_u(self.x_dm, name_u)
        self.u_val_true, grad_u_true= self.u_val, grad_u
        #
        self.v_val_u, grad_v_u= self.grad_v(self.x_dm, name_vu)
        self.v_val_a, grad_v_a= self.grad_v(self.x_dm, name_va)
        w_val, grad_w= self.fun_w(self.x_dm, self.low, self.up)
        u_bd_pred, grad_u_bd= self.grad_u(self.x_bd, name_u)
        #
        a_bd_pred= self.net_a(self.x_bd, 1, name_a, tf.AUTO_REUSE)
        #**********************************************************************
        wv_val_u= tf.multiply(w_val, self.v_val_u)
        wv_val_a= tf.multiply(w_val, self.v_val_a)
        #
        dudw_val= tf.reduce_sum(tf.multiply(grad_u, grad_w), axis=1)
        dudw_val= tf.reshape(dudw_val, [-1,1])
        dudw_val_a= tf.reduce_sum(tf.multiply(grad_u_true, grad_w), axis=1)
        dudw_val_a= tf.reshape(dudw_val_a, [-1,1])
        #
        dudv_val= tf.reduce_sum(tf.multiply(grad_u, grad_v_u), axis=1)
        dudv_val= tf.reshape(dudv_val, [-1,1])
        dudv_val_a= tf.reduce_sum(tf.multiply(grad_u_true, grad_v_a), axis=1)
        dudv_val_a= tf.reshape(dudv_val_a, [-1,1])
        #
        dudwv_val= tf.add(tf.multiply(self.v_val_u, dudw_val),
                          tf.multiply(w_val, dudv_val))
        dudwv_val_a= tf.add(tf.multiply(self.v_val_a, dudw_val_a),
                          tf.multiply(w_val, dudv_val_a))
        #
        _, _, g_obv= self.fun_g(self.x_bd, self.n_vec)
        g_val= tf.reduce_sum(tf.multiply(grad_u_bd, self.n_vec), axis=1)
        g_val= tf.reshape(g_val, [-1,1]) 
        #**********************************************************************
        with tf.variable_scope('loss'):
            with tf.name_scope('loss_u'):
                test_norm_u = tf.multiply(tf.reduce_mean(wv_val_u**2), self.int_dm)  # w*v_u 的l_2范数(v_u表示关于u的test function)
                test_norm_a = tf.multiply(tf.reduce_mean(wv_val_a**2), self.int_dm)  # w*v_a 的l_2范数(v_a表示关于a的test function)
                #******************************************************************
                # operator-norm (a(x)固定，学习u(x))
                int_r1= tf.multiply(tf.reduce_mean(tf.multiply(self.a_val, dudwv_val)), self.int_dm)
                int_l1= tf.multiply(tf.reduce_mean(tf.multiply(self.f_val, wv_val_u)), self.int_dm)
                self.loss_int= (self.beta_u)*tf.square(int_l1-int_r1) / test_norm_u
                #*******************************************************************
                # operator-norm (u(x)固定，学习a(x))
                int_r1_a= tf.multiply(tf.reduce_mean(tf.multiply(self.a_val, dudwv_val_a)), self.int_dm)
                int_l1_a= tf.multiply(tf.reduce_mean(tf.multiply(self.f_val, wv_val_a)), self.int_dm)
                self.loss_int_a= (self.beta_a)*tf.square(int_l1_a-int_r1_a) / test_norm_a  
                #
                self.loss_u_bd= tf.reduce_mean(tf.abs(u_bd_pred-self.u_bd))  # loss on boundary for u(x)
                self.loss_g_bd= tf.reduce_mean(tf.abs(g_val - g_obv))
                #
                self.loss_a_bd= tf.reduce_mean(tf.abs(a_bd_pred-self.a_bd))  # loss on boundary for a(x)
                #
                self.loss_u= (10000)*(self.loss_u_bd+self.loss_g_bd)+self.loss_int  # loss in domain for u(x)
                #
                self.loss_a= (10000)*self.loss_a_bd+self.loss_int_a    # loss in domain for a(x)
            with tf.name_scope('loss_v'):
                # 
                self.loss_v_u=  - tf.log(self.loss_int)                      # loss for v_u
                self.loss_v_a=  - tf.log(self.loss_int_a)                    # loss for v_a
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
                    self.loss_a, var_list= a_vars)
            self.v_opt_a= tf.train.AdagradOptimizer(self.v_rate_a).minimize(
                    self.loss_v_a, var_list= v_vars_a)
    
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
                            self.f_val: train_data[3],
                            self.u_bd: train_data[4],
                            self.a_bd: train_data[5],
                            self.n_vec: train_data[6]}
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
                    loss_u, loss_v, int_a, loss_a_bd= sess.run(
                        [self.loss_u, self.loss_v_a, self.loss_int_a, self.loss_a_bd], 
                        feed_dict= feed_train)
                    print('Iterations:{}'.format(i))
                    print('u_loss:{} v_loss:{} loss_a_int:{} loss_a_bd:{} l2r_a:{} l2r_u:{}'.format(
                        loss_u, loss_v, int_a, loss_a_bd, error_a[-1], error_u[-1]))
                    #
                    pred_u_draw, pred_a_draw, pred_v_draw= sess.run(
                            [self.u_val, self.a_val, self.v_val_a], 
                            feed_dict={self.x_dm: draw_x})
                    #visual.show_error(step, error_u, self.dim, 'l2r_u')
                    #visual.show_error(step, error_a, self.dim, 'l2r_a')
                    #visual.show_u_val(mesh, draw_a, pred_a_draw, 'a',  i)
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
            #visual.show_error_abs(mesh, draw_x, np.abs(draw_a-pred_a_draw), 'a', self.dim)
            #visual.show_error_abs(mesh, draw_x, np.abs(draw_u-pred_u_draw), 'u', self.dim)
            print('L2r_a is {}, L2r_u is {}'.format(np.min(error_a), np.min(error_u)))
        return(mesh, test_x, draw_x, test_u, draw_u, test_a, draw_a, pred_u, pred_u_draw, pred_a, pred_a_draw, 
               step, error_a, error_u, time_list, iter_time_list, self.dim)

if __name__=='__main__':
    dim, beta_u, beta_a, N_dm, N_bd, noise= 5, 10000, 10000, 200000, 50, 0
    noise_level= noise/100
    file_name= './problem_piecewise_1/'
    demo= wan_inv(file_name, dim, beta_u, beta_a, N_dm, N_bd, noise_level)
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
    scipy.io.savemat(file_name+'iwan_%dd_n%d'%(dim, noise), data_save)




