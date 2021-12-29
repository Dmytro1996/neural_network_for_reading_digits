/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.Arrays;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author dmytr
 */
public class ConvAltLayer extends ConvLayer {
    
    
    private INDArray input;
    private INDArray[] parsedImage;

    public ConvAltLayer(Neuron neuron) {
        super(neuron);
    }

    public ConvAltLayer(int[] image_shape, int numOfFilters, int[] kernel, Neuron neuron) {
        super(image_shape, numOfFilters, kernel, neuron);
        input=Nd4j.ones(image_shape).castTo(DataType.DOUBLE);
        setParsedImage();
    }  
    
    public INDArray feedforward(INDArray activations){
        input.muli(0).addi(activations.reshape(getImage_shape()));
        INDArray mediator=Nd4j.zeros(getNumOfFilters(), getHeight(), getWidth(),getKernel()[0],getKernel()[1]).castTo(DataType.DOUBLE);
        mediator.data().assign((DataBuffer[])Arrays.asList(parsedImage).parallelStream().map(p->p.dup().data()).toArray(size->new DataBuffer[size]));
        setZ(Nd4j.toFlattened(mediator.mul(getWeights()).sum(3,4)).add(getBiases()));
        return getActivations();
    }
    
    public void setParsedImage(){
        parsedImage=new INDArray[getNumOfFilters()*getHeight()*getWidth()];
        INDArray temp=input;
        IntStream.range(0, getNumOfFilters()).forEach(filter->
                IntStream.range(0, getHeight()).forEach(row->
                        IntStream.range(0, getWidth()).forEach(col->{
                            int currentFilter=temp.shape()[0]==1?0:filter;
                    parsedImage[filter*getHeight()*getWidth()+row*getHeight()+col]=temp.get(new INDArrayIndex[]{
                        NDArrayIndex.interval(currentFilter,currentFilter+1),
                        NDArrayIndex.interval(row,row+getKernel()[0]),
                        NDArrayIndex.interval(col,col+getKernel()[1])
                    });
        })));
    }

    public INDArray getInput() {
        return input;
    }

    public INDArray[] getParsedImage() {
        return parsedImage;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }

    public void setParsedImage(INDArray[] parsedImage) {
        this.parsedImage = parsedImage;
    }
}
