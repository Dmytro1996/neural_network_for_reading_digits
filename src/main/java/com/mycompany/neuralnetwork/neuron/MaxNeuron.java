/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.neuron;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author dmytr
 */
public class MaxNeuron implements Neuron {
    
    public INDArray fun(INDArray input){
        return Transforms.max(input, 0);
    }
    
    public INDArray derivative(INDArray input){
        return Nd4j.create(Arrays.stream(Transforms.max(input, 0).data().asDouble())
                .map(i->i>0?1:0).toArray(),input.shape(), input.dataType());
    }
}
