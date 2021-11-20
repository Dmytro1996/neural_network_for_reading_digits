/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.optimizers;

import com.mycompany.neuralnetwork.network.NetworkEnhanced;
import java.util.LinkedList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public interface Optimizer {
    
    void optimize(NetworkEnhanced net, List<INDArray[]> train_data, int epochs, int mini_batch_size, double eta,
            double lambda, boolean monitor_eval_cost, boolean monitor_train_cost,
            boolean monitor_train_acc, List<INDArray[]>...eval_data);
    
    void update_mini_batch(NetworkEnhanced net, List<INDArray[]> mini_batch, double eta, double lambda, int lenTrainData);
    
    public default INDArray[][] backprop(NetworkEnhanced net, INDArray x,INDArray y){
        INDArray[] nabla_b=new INDArray[net.getNum_layers()-1];
        INDArray [] nabla_w=new INDArray[net.getNum_layers()-1];
        INDArray activation=x;
        List<INDArray> activations=new LinkedList<>();
        activations.add(x);
        List<INDArray> zs=new LinkedList<>();
        for(int i=0;i<net.getNum_layers()-1;i++){
            //System.out.println(weights.get(i));
            //System.out.println(biases.get(i));
            zs.add(net.getWeights().get(i).mmul(activation).add(net.getBiases().get(i)));
            //System.out.println("z:"+zs.get(i));
            activation=net.getNeuron().fun(zs.get(i));
            //System.out.println("Activation:"+activation);
            activations.add(activation);
        }
        //System.out.println("Activations:"+activations.get(activations.size()-1));
        INDArray delta=net.getCost().delta(activations.get(activations.size()-1),y, null);
        //System.out.println("delta:"+delta);
        nabla_b[nabla_b.length-1]=delta.dup();
        //System.out.println(nabla_b[nabla_b.length-1]==delta);
        //System.out.println("nabla_b[nabla_b.length-1]:"+nabla_b[nabla_b.length-1]);
        nabla_w[nabla_w.length-1]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-2).reshape(
                        new int[]{1,(int)activations.get(activations.size()-2).shape()[0]}));
        //System.out.println("nabla_w[nabla_w.length-1]:"+nabla_w[nabla_w.length-1]);
        for(int i=2;i<net.getNum_layers();i++){
            int j=i;
            INDArray z=zs.get(zs.size()-i);
            INDArray sp=net.getNeuron().derivative(z);
            delta=net.getWeights().get(net.getWeights().size()-i+1).transpose().mmul(delta).mul(sp);
            //System.out.println("delta"+i+":"+delta);
            nabla_b[nabla_b.length-i]=delta.dup();
            //System.out.println("nabla_b[nabla_b.length-i]:"+nabla_b[nabla_b.length-i]);
            nabla_w[nabla_w.length-i]=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(activations.get(activations.size()-j-1).reshape(
                        new int[]{1,(int)activations.get(activations.size()-j-1).shape()[0]}));
            //System.out.println("nabla_w[nabla_w.length-i]:"+nabla_w[nabla_w.length-i]);
        }
        return new INDArray[][]{nabla_b,nabla_w};
    }
}
