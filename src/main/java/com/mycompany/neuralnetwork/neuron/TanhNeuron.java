/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.neuron;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author dmytr
 */
public class TanhNeuron implements Neuron {
    
    
    public INDArray fun(INDArray input){
        return Transforms.tanh(input,true);
    }
    
    public INDArray derivative(INDArray input){
        return Nd4j.ones(input.shape()).sub(Transforms.pow(Transforms.tanh(input,true),2));
    }
}
