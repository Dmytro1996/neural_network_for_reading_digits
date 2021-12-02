/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.cost.Cost;
import com.mycompany.neuralnetwork.neuron.Neuron;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author dmytr
 */
public class OutputLayer extends FlatLayer {
    
    private Cost cost;

    public OutputLayer(int nIn, int nOut, Neuron neuron, Cost cost) {
        super(nIn, nOut, neuron);
        this.cost=cost;
    }
    
    public INDArray[] backProp(INDArray y, INDArray prevActivations, INDArray nextDelta){
        INDArray delta=cost.delta(getNeuron().fun(getZ()),y, null);
        INDArray nabla_w=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(prevActivations.reshape(new int[]{1,(int)prevActivations.shape()[0]}));
        return new INDArray[]{delta, nabla_w};
    }
}
