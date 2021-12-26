/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.optimizers;

import com.mycompany.neuralnetwork.network.NetworkEnhanced;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDMath;

/**
 *
 * @author dmytr
 */
public class Adagrad implements Optimizer {
    
    private INDArray[][] grads_squared;
    

    public void optimize(NetworkEnhanced net, List<INDArray[]> train_data, int epochs, int mini_batch_size, double eta,
            double lambda, boolean monitor_eval_cost, boolean monitor_train_cost,
            boolean monitor_train_acc, List<INDArray[]>...eval_data){        
        int len_eval=(eval_data.length!=0)?eval_data[0].size():0;        
        int len_train=train_data.size();
        grads_squared=new INDArray[][]{net.getBiases().stream().map(b->Nd4j.zeros(b.shape())).toArray(INDArray[]::new),
                    net.getWeights().stream().map(w->Nd4j.zeros(w.shape())).toArray(INDArray[]::new)};        
        for(int i=0;i<epochs;i++){
            Collections.shuffle(train_data);
            List<List<INDArray[]>> mini_batches=IntStream.range(0,(len_train/mini_batch_size))
                    .boxed().map(c->train_data.subList(c*mini_batch_size, (c+1)*mini_batch_size))
                    .collect(Collectors.toList());
            long beginPoint=System.currentTimeMillis();            
            mini_batches.forEach(mini_batch->{
                    update_mini_batch(net, mini_batch,eta,lambda,train_data.size());
            });            
            System.out.println("\n"+(System.currentTimeMillis()-beginPoint));            
            if(monitor_train_cost){
                System.out.println("Cost on training data: {"+net.countTotalCost(train_data, lambda)+"}");
            }
            if(monitor_train_acc){
                System.out.println("Accuracy on training data: {"+net.countAccuracy(train_data, true)+"} // {"+len_train+"}");
            }
            if(monitor_eval_cost){
                System.out.println("Cost on evaluation data: {"+net.countTotalCost(eval_data[0], lambda)+"}");
            }
            if(len_eval>0){
                System.out.println("Epoch {"+i+"}: {"+net.countAccuracy(eval_data[0],false)+"} // {"+len_eval+"}");
            }
            else{
                System.out.println("Epoch {"+i+"}: completed");
            }
        }
    }

    public void update_mini_batch(NetworkEnhanced net, List<INDArray[]> mini_batch, double eta, double lambda, int lenTrainData){
        INDArray[][] nablas=mini_batch.parallelStream().map(mb->backprop(net, mb[0],mb[1]))
                .reduce((acc,x)->{
            for(int i=0;i<net.getNum_layers()-1;i++){                
                acc[0][i]=acc[0][i].add(x[0][i]);
                acc[1][i]=acc[1][i].add(x[1][i]);
            }
            return acc;
        }).get();
        for(int i=0;i<net.getNum_layers()-1;i++){
            nablas[1][i]=nablas[1][i].div(mini_batch.size());
            nablas[0][i]=nablas[0][i].div(mini_batch.size());
            grads_squared[0][i].add(nablas[0][i].mul(nablas[0][i]));
            grads_squared[1][i].add(nablas[1][i].mul(nablas[1][i]));
            net.getWeights().set(i, net.getWeights().get(i).sub(new NDMath()
                            .pow(grads_squared[1][i].add(0.0000001), -0.5).mul(eta).mul(nablas[1][i])));
            net.getBiases().set(i, net.getBiases().get(i).sub(new NDMath()
                            .pow(grads_squared[0][i].add(0.0000001), -0.5).mul(eta).mul(nablas[0][i])));
        }
        System.out.print("-");
    }    
}
