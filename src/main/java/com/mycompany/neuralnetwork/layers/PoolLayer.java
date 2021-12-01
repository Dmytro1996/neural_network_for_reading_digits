/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dmytr
 */
public class PoolLayer extends ConvLayer{

    public PoolLayer(int[] image_shape,int numOfLayers, int[] kernel, Neuron neuron) {
        super(image_shape,numOfLayers, kernel, neuron);
    }
    
   public INDArray feedforward(INDArray activations){
        if(activations.shape().length>3){
            activations.reshape(getNumOfFilters(),activations.shape()[0],activations.shape()[1]);
        }
        setZ(parseImage(activations,getKernel(),false)
                .mul(getWeights()).sum(3,4).add(getBiases()));
        return getNeuron().fun(getZ());
    }
}
