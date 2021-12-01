/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.cost.LogLikehoodCost;
import com.mycompany.neuralnetwork.neuron.SoftmaxNeuron;

/**
 *
 * @author dmytr
 */
public class SoftmaxLayer extends OutputLayer {

    public SoftmaxLayer(int nIn, int nOut) {
        super(nIn, nOut, new SoftmaxNeuron(), new LogLikehoodCost());
    }
    
}
