/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.network;

import com.mycompany.neuralnetwork.layers.ConvLayer;
import com.mycompany.neuralnetwork.layers.HiddenLayer;
import com.mycompany.neuralnetwork.layers.Layer;
import com.mycompany.neuralnetwork.layers.PoolLayer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public class MultiLayerNetwork {
    
    private List<Layer> layers;
    private int numOfLayers;
    
    public MultiLayerNetwork(Layer...layers){
        this.layers=new ArrayList<>(Arrays.asList(layers));
        this.numOfLayers=layers.length;
        //setDeltasPositions();
    }
    
    public INDArray feedforward(INDArray activations){
        for(Layer layer:layers){
            //long beginPoint=System.currentTimeMillis();
            activations=layer.feedforward(activations);
            //System.out.println(System.currentTimeMillis()-beginPoint);
        }
        return activations;
    }
    
    public void SGD(List<INDArray[]> train_data, int epochs, int mini_batch_size, double eta, double lambda, List<INDArray[]>...test_data){
        int len_test=(test_data.length!=0)?test_data[0].size():0;
        int len_train=train_data.size();
        for(int i=0;i<epochs;i++){
            Collections.shuffle(train_data);
            List<List<INDArray[]>> mini_batches=IntStream.range(0,(len_train/mini_batch_size))
                    .boxed().map(c->train_data.subList(c*mini_batch_size, (c+1)*mini_batch_size))
                    .collect(Collectors.toList());
            System.out.println(mini_batches.size());
            long beginPoint=System.currentTimeMillis();
            mini_batches.forEach(mini_batch->{
                    update_mini_batch(mini_batch,eta,lambda,len_train);
            });
            System.out.println("\n"+(System.currentTimeMillis()-beginPoint));
            if(len_test>0){
                System.out.println("Epoch {"+i+"}: {"+evaluate(test_data[0])+"} // {"+len_test+"}");
            }
            else{
                System.out.println("Epoch {"+i+"}: completed");
            }
        }
    }
    
    public void update_mini_batch(List<INDArray[]> mini_batch, double eta, double lambda, int lenTrainData){
        INDArray[][] nablas=mini_batch.parallelStream().map(mb->backProp(mb[0],mb[1]))
                .reduce((acc,x)->{
            for(int i=0;i<numOfLayers-1;i++){ 
                acc[i][0]=acc[i][0].add(x[i][0]);
                acc[i][1]=acc[i][1].add(x[i][1]);
            }
            return acc;
        }).get(); 
        for(int i=0;i<numOfLayers-1;i++){
            nablas[i][1]=nablas[i][1].mul(eta/mini_batch.size());
            nablas[i][0]=nablas[i][0].mul(eta/mini_batch.size());
            ((HiddenLayer)layers.get(i+1)).setWeights(((HiddenLayer)layers.get(i+1))
                    .getWeights().sub(nablas[i][1]));
            /*((HiddenLayer)layers.get(i+1)).setWeights(((HiddenLayer)layers.get(i+1))
                    .getWeights().mul(1d-eta*Math.round(lambda/lenTrainData)).sub(nablas[i][1]));*/
            ((HiddenLayer)layers.get(i+1)).setBiases(((HiddenLayer)layers.get(i+1)).getBiases().sub(nablas[i][0]));
        }
        System.out.print("-");
    }
    
    public INDArray[][] backProp(INDArray x, INDArray y){
        INDArray[][] result=new INDArray[layers.size()-1][];
        //System.out.println("x.length:"+x.length());
        INDArray activation=x.dup();
        //long beginPoint=System.currentTimeMillis();
        for(Layer layer:layers){
            activation=layer.feedforward(activation);
        }
        //long feedforwardTime=System.currentTimeMillis()-beginPoint;
        result[result.length-1]=layers.get(layers.size()-1).backProp(y, layers.get(layers.size()-2)
                .getActivations(), null);
        for(int i=result.length-2;i>=0;i--){
            if(layers.get(i+1) instanceof ConvLayer && layers.get(i+2) instanceof ConvLayer){
                result[i]=((ConvLayer)layers.get(i+1)).backPropConv(((HiddenLayer)layers.get(i+2)).getWeights(),
                     layers.get(i).getActivations(),result[i+1][0],
                     layers.get(i+2) instanceof PoolLayer);
            } else{
                result[i]=layers.get(i+1).backProp(((HiddenLayer)layers.get(i+2)).getWeights(), layers.get(i)
                    .getActivations(), result[i+1][0]);
            }
            //System.out.println("result[i]: "+Arrays.toString(result[i]));
        }
        //long backPropTime=System.currentTimeMillis()-beginPoint;
        //System.out.println("Backpropagation: "+(backPropTime-feedforwardTime)+" Feedforward: "+feedforwardTime);
        return result;        
    }
    
    public long evaluate(List<INDArray[]> test_data){
        return test_data.parallelStream().filter(c->
                feedforward(c[0]).argMax(0).data().asInt()[0]==c[1].data().asInt()[0])
                .count();
    }    
}
