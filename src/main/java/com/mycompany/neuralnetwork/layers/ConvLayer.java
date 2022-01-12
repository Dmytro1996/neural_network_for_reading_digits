/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork.layers;

import com.mycompany.neuralnetwork.neuron.Neuron;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
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
public class ConvLayer extends HiddenLayer implements IConvLayer {
    
    private int[] image_shape;
    private int[] kernel;
    private int numOfFilters;
    private int width;
    private int height;
    
    public ConvLayer(Neuron neuron){
        super(neuron);
    }

    public ConvLayer(int[] image_shape,int numOfFilters, int[] kernel, Neuron neuron) {
        super(neuron);
        this.image_shape = image_shape;
        this.numOfFilters=numOfFilters;
        this.kernel = kernel;
        height=image_shape[1]-(kernel[0]-1);
        width=image_shape[2]-(kernel[1]-1);
        Random rand=new Random();
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(numOfFilters*kernel[0]*kernel[1]).toArray(),
                new long[]{numOfFilters,1,1,kernel[0],kernel[1]}, DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(numOfFilters*width*height).toArray(),
                new long[]{numOfFilters*width*height}, DataType.DOUBLE));
    }
    
    public INDArray feedforward(INDArray activations){
        setZ(Nd4j.toFlattened(mulConv(activations)).add(getBiases()));
        return getActivations();
    }   
        
    public INDArray[] backPropConv(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){
        INDArray delta=calculateDeltas(nextDelta, nextWeights).mul(getNeuron().derivative(getZ()));
        INDArray nabla_w=parseImage(prevActivations)//test ok
                .mul(delta.reshape(numOfFilters, height, width,1,1)).sum(1,2)
                .reshape(numOfFilters,1,1,kernel[0],kernel[1]);
        return new INDArray[]{delta.reshape(delta.length()), nabla_w};
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){
        INDArray delta=nextWeights.transpose().mmul(nextDelta).mul(getNeuron().derivative(getZ()));
        INDArray nabla_w=parseImage(prevActivations)
                .mul(delta.reshape(numOfFilters, height, width,1,1)).sum(1,2)
                .reshape(numOfFilters,1,1,kernel[0],kernel[1]);
        return new INDArray[]{delta, nabla_w};
    }
    
    public INDArray mulConv(INDArray image){
        return parseImage(image).mul(getWeights()).sum(3,4);
    }
    
    public INDArray parseImage(INDArray image){
        long beginPoint=System.currentTimeMillis();
        INDArray input=image.dup();
        if(input.length()==image_shape[1]*image_shape[2]){
            input=multiplyByFilters(input).reshape(numOfFilters,image_shape[1],image_shape[2]);
        } else{
            input=input.reshape(image_shape);
        }
        input=multiplyByRows(input);
        INDArray result=Nd4j.zeros(DataType.DOUBLE,numOfFilters,height,width,kernel[0],kernel[1]);
        INDArray temp=input;
        IntStream.range(0, width).parallel().forEach(col->{
                    result.put(new INDArrayIndex[]{
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.point(col),
                        NDArrayIndex.all(),NDArrayIndex.all()
                    },temp.get(new INDArrayIndex[]{
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(col,col+kernel[1])
                    }));
        });
        return result;
    }
    
    public INDArray multiplyByFilters(INDArray image){
        INDArray result=Nd4j.zeros(numOfFilters,image.length());
        IntStream.range(0,numOfFilters).forEach(filter->{
            result.put(new INDArrayIndex[]{NDArrayIndex.point(filter),NDArrayIndex.all()},image);
        });
        return result;
    }
    
    public INDArray multiplyByRows(INDArray image){
        INDArray result=Nd4j.zeros(numOfFilters, height, kernel[0], image_shape[2]);
        IntStream.range(0,height).forEach(row->{
            result.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(row)},
                    image.get(NDArrayIndex.all(),NDArrayIndex.interval(row,row+kernel[0])));
        });
        return result;
    }
    
    public INDArray calculateDeltas(INDArray nextDelta, INDArray nextWeights){
        INDArray delta=Nd4j.zeros(numOfFilters,height,width);
        int horStride=1;
        int vertStride=1;
        int nextLayerHeight=0;
        int nextLayerWidth=0;
        int[] nextKernel=new int[]{(int)nextWeights.shape()[3],(int)nextWeights.shape()[4]};
        if(nextDelta.length()*nextWeights.shape()[3]*nextWeights.shape()[4]==numOfFilters*height*width){
            vertStride=(int)nextWeights.shape()[3];
            horStride=(int)nextWeights.shape()[4];
            nextLayerHeight=height/vertStride;
            nextLayerWidth=width/horStride;
        } else{
            nextLayerHeight=height-((int)nextWeights.shape()[3]-1);
            nextLayerWidth=width-((int)nextWeights.shape()[4]-1);            
        }
        int nextLayerFilters=(int)nextDelta.length()/(nextLayerHeight*nextLayerWidth);
        INDArray nextDeltaReshaped=nextDelta.reshape(nextLayerFilters, 1, nextLayerHeight*nextLayerWidth);
        INDArray nextWeightsReshaped=nextWeights.reshape(nextLayerFilters,nextWeights.shape()[3],nextWeights.shape()[4]);
        int[] strides=new int[]{vertStride,horStride};
        INDArray temp=delta;
        int[] sizes=new int[]{nextLayerHeight,nextLayerWidth};
        IntStream.range(0, nextLayerHeight*nextLayerWidth).parallel().forEach(i->{
            int row=i/sizes[0];
            int col=i-row*sizes[0];
            temp.get(NDArrayIndex.all(),NDArrayIndex.interval(row*strides[0], row*strides[0]+nextKernel[0]),
                    NDArrayIndex.interval(col*strides[1], col*strides[1]+nextKernel[1]))
                    .addi(nextDeltaReshaped.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(i,i+1))
                            .mul(nextWeightsReshaped));
        });
        return Nd4j.toFlattened(delta);
    }
    
    public int[] getImage_shape() {
        return image_shape;
    }

    public int getNumOfFilters() {
        return numOfFilters;
    }

    public int[] getKernel() {
        return kernel;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public void setImage_shape(int[] image_shape) {
        this.image_shape = image_shape;
    }

    public void setNumOfFilters(int numOfFilters) {
        this.numOfFilters = numOfFilters;
    }

    public void setKernel(int[] kernel) {
        this.kernel = kernel;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }
}
