/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.cost.Cost;
import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dmytr
 */
public class ConvLayer {
    
    private int[] filter_shape;
    private int[] image_shape;
    private int[] pool_size;
    private Neuron neuron;
    private int n_in;
    private int n_out;
    private INDArray weights;
    private INDArray biases;
    private INDArray z;

    public ConvLayer(int[] filter_shape, int[] image_shape, int[] pool_size, Neuron neuron) {
        this.filter_shape = filter_shape;
        this.image_shape = image_shape;
        this.pool_size = pool_size;
        this.neuron = neuron;
        Random rand=new Random();
        weights=Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(filter_shape[2]*filter_shape[3]).toArray(),
                new long[]{filter_shape[2],filter_shape[3]}, DataType.DOUBLE);
        biases=Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(n_out).toArray(),
                new long[]{n_out}, DataType.DOUBLE);
    }
    
    public INDArray feedforward(INDArray activations){
        z=parseImage(activations).mul(weights).sum(2,3).add(biases);
        return neuron.fun(z);
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray prevDelta){
        INDArray delta=nextWeights.mmul(prevDelta).mul(neuron.derivative(z));
        INDArray nabla_w=parseImage(prevActivations).mul(delta).sum(0,1);
        return new INDArray[]{delta, nabla_w};
    }
    
    public INDArray parseImage(INDArray image){
        double[] imgArr=image.data().asDouble();
        List<Double> resultList=new ArrayList<>();
        int filter_height=filter_shape[2];
        int filter_width=filter_shape[3];
        int image_height=image_shape[2];
        int image_width=image_shape[3];
        for(int row=0;row+filter_height<=image_height;row++){
            for(int i=0;i+filter_width<=image_width;i++){
                for(int subRow=0;subRow<filter_height;subRow++){
                    for(int subCol=0;subCol<filter_width;subCol++){
                        resultList.add(imgArr[row*image_width+i+subRow*image_width+subCol]); 
                    }
                }
            }
        }
        return Nd4j.create((double[])resultList.stream().mapToDouble(d->d.doubleValue()).toArray(),
                new long[]{image_height-(filter_height-1),image_width-(filter_width-1),filter_height,filter_width},DataType.DOUBLE);
    }
    
    
}
