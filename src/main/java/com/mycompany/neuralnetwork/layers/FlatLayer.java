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
public class FlatLayer extends HiddenLayer {
    
    private int nIn;
    private int nOut;

    public FlatLayer(int nIn, int nOut, Neuron neuron) {
        super(neuron);
        this.nIn = nIn;
        this.nOut = nOut;
        Random rand=new Random();        
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()/Math.sqrt(nOut))
                .limit(nIn*nOut).toArray(),
                    new long[]{nOut,nIn},DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian())
                        .limit(nOut).toArray(),new long[]{nOut},DataType.DOUBLE));
    }
    
    public INDArray feedforward(INDArray activations){
        setZ(activations.mmul(getWeights()).add(getBiases()));
        return getActivations();
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){
        INDArray delta=nextWeights.transpose().mmul(nextDelta).mul(getNeuron().derivative(getZ()));
        INDArray nabla_w=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(prevActivations.reshape(new int[]{1,(int)prevActivations.shape()[0]}));
        return new INDArray[]{delta, nabla_w};
    }
    
    public int getnIn() {
        return nIn;
    }

    public int getnOut() {
        return nOut;
    }    

    public void setnIn(int nIn) {
        this.nIn = nIn;
    }

    public void setnOut(int nOut) {
        this.nOut = nOut;
    }        
    
}
