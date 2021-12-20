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
    private int[][] deltasPositions;
    private INDArray input;
    private INDArray[] parsedImage;
    
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
        input=Nd4j.ones(image_shape).castTo(DataType.DOUBLE);
        setParsedImage();
    }
    
    public INDArray feedforward(INDArray activations){
        setZ(Nd4j.toFlattened(mulConv(activations)).add(getBiases()));
        return getActivations();
    }
    
    public INDArray feedforwardNew(INDArray activations){
        long beginPoint=System.currentTimeMillis();
        input.muli(0).addi(activations.reshape(image_shape));
        System.out.println(System.currentTimeMillis()-beginPoint);
        /*setZ(Nd4j.toFlattened(Nd4j.create(
                parsedImage.parallelStream().map(
                        p->p.dup().data().asDouble()).flatMapToDouble(Arrays::stream).toArray(),
                new long[]{numOfFilters,height,width,kernel[0],kernel[1]},
                DataType.DOUBLE).mul(getWeights()).sum(3,4)).add(getBiases()));*/
        setZ(Nd4j.toFlattened(Nd4j.vstack(parsedImage)
                .reshape(numOfFilters,height,width,kernel[0],kernel[1]).mul(getWeights()).sum(3,4)).add(getBiases()));
        System.out.println(System.currentTimeMillis()-beginPoint);
        return getActivations();
    }
    
    public INDArray[] backPropConv(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){        
        /*double[] deltas=new double[numOfFilters*width*height];
        double[] ndArr=Nd4j.matmul(nextDelta.reshape(new long[]{numOfFilters,nextDelta.shape()[0]/numOfFilters,1})//test
                ,nextWeights.reshape(numOfFilters,1,nextWeights.shape()[3]*nextWeights.shape()[4])).data().asDouble();
        for(int i=0;i<deltas.length;i++){
            try{
            for(int j:deltasPositions[i]){
                deltas[i]+=ndArr[j];
            }
            } catch(NullPointerException e){
                System.out.println("Iteration:"+i);
                System.out.println("deltasPositions==null:"+(deltasPositions==null));
                System.out.println("deltasPositions[i]:"+Arrays.toString(deltasPositions[i]));
                throw new NullPointerException();
            }
        }
        INDArray delta=Nd4j.create(deltas,new long[]{width*height},DataType.DOUBLE);*/
        INDArray delta=Nd4j.matmul(nextDelta.reshape(new long[]{numOfFilters,nextDelta.shape()[0]/numOfFilters,1})//test
                ,nextWeights.reshape(numOfFilters,1,nextWeights.shape()[3]*nextWeights.shape()[4]));        
        INDArray nabla_w=parseImageNew(prevActivations)//test ok
                .mul(delta.reshape(numOfFilters, height, width,1,1)).sum(1,2)
                .reshape(numOfFilters,1,1,kernel[0],kernel[1]);
        return new INDArray[]{delta.reshape(delta.length()), nabla_w};
    }
    
    public INDArray[] backProp(INDArray nextWeights, INDArray prevActivations, INDArray nextDelta){        
        INDArray delta=nextWeights.transpose().mmul(nextDelta).mul(getNeuron().derivative(getZ()));
        INDArray nabla_w=parseImageNew(prevActivations)
                .mul(delta.reshape(numOfFilters, height, width,1,1)).sum(1,2)
                .reshape(numOfFilters,1,1,kernel[0],kernel[1]);
        return new INDArray[]{delta, nabla_w};
    }
    
    public INDArray mulConv(INDArray image){
        return parseImageNew(image).mul(getWeights()).sum(3,4);
    }
    
    public INDArray parseImageNew(INDArray image){
        long beginPoint=System.currentTimeMillis();
        INDArray input=image.dup();
        if(input.length()==image_shape[1]*image_shape[2]){
            input=input.reshape(1,image_shape[1],image_shape[2]);
        } else{
            input=input.reshape(image_shape);
        }
        INDArray result=Nd4j.zeros(DataType.DOUBLE,numOfFilters,height,width,kernel[0],kernel[1]);
        INDArray temp=input;
        IntStream.range(0, numOfFilters).parallel().forEach(filter->
                IntStream.range(0, height).parallel().forEach(row->
                        IntStream.range(0, width).parallel().forEach(col->{
                            int currentFilter=temp.shape()[0]==1?0:filter;
                    result.put(new INDArrayIndex[]{
                        NDArrayIndex.interval(filter,filter+1),
                        NDArrayIndex.point(row),
                        NDArrayIndex.point(col),
                        NDArrayIndex.all(),NDArrayIndex.all()
                    },temp.get(new INDArrayIndex[]{
                        NDArrayIndex.interval(currentFilter,currentFilter+1),
                        //NDArrayIndex.all(),NDArrayIndex.all(),
                        NDArrayIndex.interval(row,row+kernel[0]),
                        NDArrayIndex.interval(col,col+kernel[1])
                    }));
                        })));
        System.out.println("Parese image: "+(System.currentTimeMillis()-beginPoint));
        return result;
    }
    
    public void setParsedImage(){
        parsedImage=new INDArray[numOfFilters*height*width];
        INDArray temp=input;
        IntStream.range(0, numOfFilters).forEach(filter->
                IntStream.range(0, height).forEach(row->
                        IntStream.range(0, width).forEach(col->{
                            int currentFilter=temp.shape()[0]==1?0:filter;
                    parsedImage[filter*height*width+row*height+col]=temp.get(new INDArrayIndex[]{
                        NDArrayIndex.interval(currentFilter,currentFilter+1),
                        NDArrayIndex.interval(row,row+kernel[0]),
                        NDArrayIndex.interval(col,col+kernel[1])
                    });
        })));
    }
    
    public INDArray parseImage(INDArray image,int[] image_shape, int[] kernel, boolean isStrideOne){
        double[] imgArr=image.data().asDouble();
        List<Double> resultList=new ArrayList<>();
        int kernel_height=kernel[0];
        int kernel_width=kernel[1];
        int image_height=(int)image_shape[1];
        int image_width=(int)image_shape[2];
        int numOfFilters=image_shape[0];
        int[] stride=isStrideOne?new int[]{1,1}:kernel;
        //System.out.println("Image length:"+image.length());
        for(int filter=0;filter<numOfFilters;filter++){
            for(int row=0;row+kernel_height<=image_height;row+=stride[0]){
                for(int col=0;col+kernel_width<=image_width;col+=stride[1]){
                    for(int subRow=0;subRow<kernel_height;subRow++){
                        for(int subCol=0;subCol<kernel_width;subCol++){
                            if(image.length()!=numOfFilters*image_height*image_width && image.length()==image_height*image_width){
                                resultList.add(imgArr[row*image_width+col+subRow*image_width+subCol]);
                            } else{
                                resultList.add(imgArr[filter*image_height*image_width
                                        +row*image_width+col+subRow*image_width+subCol]);
                            }
                        }
                    }
                }
            }
        }
        //System.out.println("result size:"+resultList.size());
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
        positions=parseImage(positions,new int[]{numOfFilters, height, width}, kernel,isStrideOne);
        int[][] result=new int[height*width][];
        List<Integer> subRes=new ArrayList<>();
        System.out.println("getDeltasPositions before loop");
        System.out.println("getDeltasPositions positions.length:"+positions.length());
        for(int i=0;i<result.length;i++){
            //int pos=0;
            for(int j=0;j<positions.data().asDouble().length;j++){
                if(positions.data().asDouble()[j]==i){                    
                    //result[i][pos]=j;
                    //pos++;
                    subRes.add(j/(kernel[0]*kernel[1]));
                }
            }
            System.out.print("-");
            result[i]=subRes.stream().mapToInt(n->n).toArray();
            subRes.clear();
        }
        System.out.println("getDeltasPositions ended");
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

    public int[][] getDeltasPositions() {
        return deltasPositions;
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

    public void setDeltasPositions(int[][] deltasPositions) {
        this.deltasPositions = deltasPositions;
    }
    
}
