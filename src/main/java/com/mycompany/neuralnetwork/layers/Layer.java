/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public interface Layer {
    
    INDArray feedforward(INDArray activations);
    INDArray[] backProp(INDArray y, INDArray prevActivations, INDArray nextDelta);
    INDArray getActivations();
}
