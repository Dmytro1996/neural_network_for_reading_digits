/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
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
public class PoolAltLayer extends ConvAltLayer {
    
    public PoolAltLayer(int[] image_shape,int numOfFilters, int[] kernel, Neuron neuron){
        super(neuron);
        setImage_shape(image_shape);
        setNumOfFilters(numOfFilters);
        setKernel(kernel);
        setHeight(image_shape[1]/kernel[0]);
        setWidth(image_shape[2]/kernel[1]);
        Random rand=new Random();
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(numOfFilters*kernel[0]*kernel[1]).toArray(),
                new long[]{numOfFilters,1,1,kernel[0],kernel[1]}, DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(getNumOfFilters()*getWidth()*getHeight()).toArray(),
                new long[]{numOfFilters*getWidth()*getHeight()}, DataType.DOUBLE));
        setInput(Nd4j.ones(image_shape).castTo(DataType.DOUBLE));
        setParsedImage();
    }
    
    public INDArray parseImageNew(INDArray image){
        INDArray input=image.dup();
        if(input.length()==getImage_shape()[1]*getImage_shape()[2]){
            input=input.reshape(1,getImage_shape()[1],getImage_shape()[2]);
        } else{
            input=input.reshape(getImage_shape());
        }
        INDArray result=Nd4j.zeros(DataType.DOUBLE,getNumOfFilters(),getHeight(),getWidth(),getKernel()[0],getKernel()[1]);
        INDArray temp=input;
        IntStream.range(0, getNumOfFilters()).parallel().forEach(filter->
                IntStream.range(0, getHeight()).parallel().forEach(row->{
                        int inputRow=row*getKernel()[0];
                        IntStream.range(0, getWidth()).parallel().forEach(col->{
                            int currentFilter=temp.shape()[0]==1?0:filter;
                            int inputCol=col*getKernel()[1];
                            result.put(new INDArrayIndex[]{
                                NDArrayIndex.interval(filter,filter+1),
                                NDArrayIndex.point(row),
                                NDArrayIndex.point(col),
                                NDArrayIndex.all(),NDArrayIndex.all()
                                },temp.get(new INDArrayIndex[]{
                                NDArrayIndex.interval(currentFilter,currentFilter+1),
                                NDArrayIndex.interval(inputRow,inputRow+getKernel()[0]),
                                NDArrayIndex.interval(inputCol,inputCol+getKernel()[1])
                            }));
                        });}));
        return result;
    }
    
    public void setParsedImage(){
        INDArray[] parsedImage=new INDArray[getNumOfFilters()*getHeight()*getWidth()];
        INDArray temp=getInput();
        IntStream.range(0, getNumOfFilters()).forEach(filter->
                IntStream.range(0, getHeight()).forEach(row->{
                        int inputRow=row*getKernel()[0];
                        IntStream.range(0, getWidth()).forEach(col->{
                            int inputCol=col*getKernel()[1];
                            int currentFilter=temp.shape()[0]==1?0:filter;
                        parsedImage[filter*getHeight()*getWidth()+row*getHeight()+col]=temp.get(new INDArrayIndex[]{
                            NDArrayIndex.interval(currentFilter,currentFilter+1),
                            NDArrayIndex.interval(inputRow,inputRow+getKernel()[0]),
                            NDArrayIndex.interval(inputCol,inputCol+getKernel()[1])
                        });
        });}));
        setParsedImage(parsedImage);
    }
}
