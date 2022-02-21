/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author dmytr
 */
public class PoolLayer extends ConvLayer{

    public PoolLayer(int[] image_shape,int numOfFilters, int[] kernel, Neuron neuron) {
        super(neuron);
        setImage_shape(image_shape);
        setNumOfFilters(numOfFilters);
        setKernel(kernel);
        setHeight(image_shape[1]/kernel[0]);
        setWidth(image_shape[2]/kernel[1]);
        Random rand=new Random();
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(numOfFilters*kernel[0]*kernel[1]).toArray(),
                new long[]{numOfFilters,1,1,kernel[0],kernel[1]}, DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(getNumOfFilters()).toArray(),
                new long[]{numOfFilters}, DataType.DOUBLE));
    }
    
    public INDArray mulConv(INDArray image){
        return parseImage(image).mul(getWeights()).max(3,4);
    }
    
    public INDArray parseImage(INDArray image){
        INDArray input=image.dup();
        if(input.length()==getImage_shape()[1]*getImage_shape()[2]){
            input=multiplyByFilters(input).reshape(getNumOfFilters(),getImage_shape()[1],getImage_shape()[2]);
        } else{
            input=input.reshape(getImage_shape());
        }
        input=multiplyByRows(input);
        INDArray result=Nd4j.zeros(DataType.DOUBLE,getNumOfFilters(),getHeight(),getWidth(),getKernel()[0],getKernel()[1]);
        INDArray temp=input;
        IntStream.range(0, getWidth()).parallel().forEach(col->{
                int inputCol=col*getKernel()[1];
                result.put(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.point(col),
                    NDArrayIndex.all(),NDArrayIndex.all()
                    },temp.get(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(inputCol,inputCol+getKernel()[1])
                    }));
            });
        return result;
    }
    
    public INDArray multiplyByRows(INDArray image){
        INDArray result=Nd4j.zeros(getNumOfFilters(), getHeight(), getKernel()[0], getImage_shape()[2]);
        IntStream.range(0,getHeight()).forEach(row->{
            int imageRow=row*getKernel()[0];
            result.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(row)},
                    image.get(NDArrayIndex.all(),NDArrayIndex.interval(imageRow,imageRow+getKernel()[0])));
        });
        return result;
    }
}
