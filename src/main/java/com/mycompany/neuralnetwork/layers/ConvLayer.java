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

/**
 *
 * @author dmytr
 */
public class ConvLayer extends HiddenLayer implements IConvLayer {
    
    private int[] image_shape;
    private int[] kernel;
    private int numOfFilters;
    int width;
    int height;

    public ConvLayer(int[] image_shape,int numOfFilters, int[] kernel, Neuron neuron) {
        super(neuron);
        this.image_shape = image_shape;
        this.numOfFilters=numOfFilters;
        this.kernel = kernel;
        height=image_shape[2]-(kernel[0]-1);
        width=image_shape[3]-(kernel[1]-1);
        Random rand=new Random();
        setWeights(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(kernel[0]*kernel[1]).toArray(),
                new long[]{numOfFilters,kernel[0],kernel[1]}, DataType.DOUBLE));
        setBiases(Nd4j.create(DoubleStream.generate(()->rand.nextGaussian()).limit(width*height).toArray(),
                new long[]{width*height}, DataType.DOUBLE));
    }
    
    public INDArray feedforward(INDArray activations){
        if(activations.shape().length>3){
            activations.reshape(numOfFilters,activations.shape()[0],activations.shape()[1]);
        }
        setZ(parseImage(activations,kernel,true)
                .mul(getWeights()).sum(3,4).add(getBiases()));
        return getActivations();
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta, int[][] deltasPositions){        
        double[] deltas=new double[width*height];
        double[] ndArr=nextDelta.reshape(new long[]{nextDelta.shape()[0],1})
                .mmul(nextWeights.reshape(1,nextWeights.shape()[0])).data().asDouble();
        for(int i=0;i<deltas.length;i++){
            for(int j:deltasPositions[i]){
                deltas[i]+=ndArr[j];
            }
        }
        INDArray delta=Nd4j.create(deltas,new long[]{width*height},DataType.DOUBLE);        
        INDArray nabla_w=parseImage(prevActivations.reshape(numOfFilters,prevActivations.shape()[0],
                prevActivations.shape()[1]),kernel,true).mul(delta).sum(1,2);
        return new INDArray[]{delta, nabla_w};
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){        
        INDArray delta=nextWeights.transpose().mmul(nextDelta).mul(getNeuron().derivative(getZ()));
        INDArray nabla_w=delta.reshape(new int[]{(int)delta.shape()[0],1})
                .mmul(prevActivations.reshape(
                        new int[]{1,(int)prevActivations.shape()[0]}));
        return new INDArray[]{delta, nabla_w};
    }
    
    public INDArray parseImage(INDArray image,int[] kernel, boolean isStrideOne){
        double[] imgArr=image.data().asDouble();
        List<Double> resultList=new ArrayList<>();
        int kernel_height=kernel[0];
        int kernel_width=kernel[1];
        int image_height=(int)image.shape()[1];
        int image_width=(int)image.shape()[2];
        int numOfFilters=(int)image.shape()[0];
        int[] stride=isStrideOne?new int[]{1,1}:kernel;
        for(int filter=0;filter<numOfFilters;filter+=image.length()/numOfFilters){
            for(int row=0;row+kernel_height<=image_height;row+=stride[0]){
                for(int col=0;col+kernel_width<=image_width;col+=stride[1]){
                    for(int subRow=0;subRow<kernel_height;subRow++){
                        for(int subCol=0;subCol<kernel_width;subCol++){
                            resultList.add(imgArr[filter+row*image_width+col+subRow*image_width+subCol]); 
                        }
                    }
                }
            }
        }
        if(!isStrideOne){
            return Nd4j.create((double[])resultList.stream().mapToDouble(d->d.doubleValue()).toArray(),
                new long[]{numOfFilters,image_height/kernel_height,image_width/kernel_width,kernel_height,kernel_width},DataType.DOUBLE);
        }
        return Nd4j.create((double[])resultList.stream().mapToDouble(d->d.doubleValue()).toArray(),
                new long[]{numOfFilters,image_height-(kernel_height-1),image_width-(kernel_width-1),kernel_height,kernel_width},DataType.DOUBLE);
    }
    
    public int[][] getDeltasPositions(int[] kernel, boolean isStrideOne){
        INDArray positions=Nd4j.create(IntStream.range(0, height*width).toArray(),
                new long[]{1,height,width},DataType.INT64);
        positions=parseImage(positions,kernel,isStrideOne);
        int[][] result=new int[height*width][];
        List<Integer> subRes=new ArrayList<>();
        for(int i=0;i<result.length;i++){
            //int pos=0;
            for(int j=0;j<positions.data().asDouble().length;j++){
                if(positions.data().asDouble()[j]==i){
                    //result[i][pos]=j;
                    //pos++;
                    subRes.add(j/(kernel[0]*kernel[1]));
                }
            }
            result[i]=subRes.stream().mapToInt(n->n).toArray();
            subRes.clear();
        }
        return result;
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
