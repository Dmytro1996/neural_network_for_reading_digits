/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dmytr
 */
public class PoolLayer extends ConvLayer{

    public PoolLayer(int[] image_shape,int numOfFilters, int[] kernel, Neuron neuron) {
        super(neuron);
        setImage_shape(image_shape);
        setNumOfFilters(numOfFilters);
        setKernel(kernel);
        setHeight(image_shape[1]/kernel[0]);
        setWidth(image_shape[2]/kernel[1]);
        Random rand=new Random();
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(numOfFilters*kernel[0]*kernel[1]).toArray(),
                new long[]{numOfFilters,1,1,kernel[0],kernel[1]}, DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(getNumOfFilters()*getWidth()*getHeight()).toArray(),
                new long[]{numOfFilters*getWidth()*getHeight()}, DataType.DOUBLE));
    }
    
   public INDArray feedforward(INDArray activations){
        /*if(activations.shape().length>3){
            activations.reshape(getNumOfFilters(),activations.shape()[0],activations.shape()[1]);
        }*/
         setZ(parseImage(activations,getImage_shape(),getKernel(),false)
                .mul(getWeights()).sum(3,4));//test ok
        setZ(getZ().add(getBiases()));
        return getActivations();
    }
}
