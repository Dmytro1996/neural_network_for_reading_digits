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
public class InputLayer implements Layer {
    
    INDArray activations;
    
    public InputLayer() {
    }
    
    public INDArray feedforward(INDArray activations){
        this.activations=activations;
        return activations;
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){        
        return null;
    } 
    
    public INDArray getActivations(){
        return activations;
    }
}
