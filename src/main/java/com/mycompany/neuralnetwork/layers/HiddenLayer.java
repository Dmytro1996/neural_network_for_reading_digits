/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.Random;
import java.util.stream.DoubleStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author dmytr
 */
public abstract class HiddenLayer implements Layer {
    
    private Neuron neuron;
    private INDArray z;
    private INDArray weights;
    private INDArray biases;

    public HiddenLayer(Neuron neuron) {
        this.neuron = neuron;
    }
    
    public INDArray getActivations(){
        return neuron.fun(z);
    }
        
    public Neuron getNeuron() {
        return neuron;
    }

    public INDArray getZ() {
        return z;
    }

    public INDArray getWeights() {
        return weights;
    }

    public INDArray getBiases() {
        return biases;
    }
    
    public void setNeuron(Neuron neuron) {
        this.neuron = neuron;
    }

    public void setZ(INDArray z) {
        if(z.shape().length>1){
            this.z = z.reshape(z.length());
        } else{
            this.z=z;
        }
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public void setBiases(INDArray biases) {
        this.biases = biases;
    }
}
