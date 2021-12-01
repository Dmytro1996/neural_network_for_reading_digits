/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.neuron;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author dmytr
 */
public class SoftmaxNeuron implements Neuron {
    
    public INDArray fun(INDArray input){
        return Transforms.softmax(input, true);
    }
    
    public INDArray derivative(INDArray input){
        return Transforms.softmax(input,true).mul(Transforms.softmax(input,true).mul(-1).add(1));
    }
}
