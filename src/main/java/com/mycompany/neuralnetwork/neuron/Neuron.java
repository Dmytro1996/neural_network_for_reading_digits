/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.neuron;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public interface Neuron {
    
    INDArray fun(INDArray input);
    
    INDArray derivative(INDArray input);
}
