/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuralnetwork;

import com.mycompany.neuralnetwork.exceptions.MatrixException;
import com.mycompany.neuralnetwork.matrix.Matrix;
import com.mycompany.neuralnetwork.matrix.Matrices;
import com.mycompany.neuralnetwork.network.Network;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import com.google.gson.Gson;
import com.mycompany.neuralnetwork.dataloader.DataLoader;
import com.mycompany.neuralnetwork.network.NetworkNd4j;
import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author dmytr
 */
public class NeuralNetwork {
    
    public static void main(String[] args) throws MatrixException, Exception{
        Matrix matrix=new Matrix(new int[]{9},Integer.class);
        Matrix newMatrix=new Matrix(new int[]{9});
        //List<Number> matrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,9,10,11,12});
        List<Number> matrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,7});
        matrix.setElements(matrixElems);        
        List<Number> newMatrixElems=Arrays.asList(new Number[]{1,2,3,4,5,6,7,8,9});
        newMatrix.setElements(newMatrixElems);
        System.out.println("Matrix:"+matrix.getElements().size());
        System.out.println(matrix.toString());
        System.out.println("newMatrix:");
        System.out.println(newMatrix.toString());
        //matrix.add(newMatrix);
        //matrix.applyFunc((e)->e=e.doubleValue()+2);
        //matrix.transpose();
        //matrix.add(1d);
        //matrix.matrixMult(newMatrix);
        //System.out.println("Matrixes addition:"+Matrixes.add(matrix,newMatrix));
        System.out.println("Result:");
        System.out.println(matrix.toString());
        //System.out.println(Matrices.applyFunc(matrix, (i)->i*2));
        //System.out.println(matrix.sum(0).toString());
        System.out.println("Combine lists result:");
        //System.out.println(Matrixes.combineElements(Arrays.asList(new Integer[]{1,2,3,4,5}),Arrays.asList(new Integer[]{11,12,30,24,50})));
        Matrix input=new Matrix(Arrays.asList(new Integer[]{1,1,1}),new int[]{3},Double.class);
        Matrix output=new Matrix(Arrays.asList(new Integer[]{1,1}),new int[]{2},Double.class);
        //Network net=new Network(new int[]{784,30,10});
        NetworkNd4j net=new NetworkNd4j(new int[]{784,30,10});
        /*System.out.println("Feedforward:"+
                net.feedforward(new Matrix(Arrays.asList(new Integer[]{1,0,1}),new int[]{3},Double.class)));
        System.out.println("Sigmoid:"+net.sigmoid(10));
        System.out.println("Sisgmoid_prime:"+net.sigmoid_prime(10));
        System.out.println("Cost_derivative:"+net.cost_derivative(matrix,newMatrix));
        System.out.println("Evaluate:"+net.evaluate(Arrays.asList(new Matrix[][]{new Matrix[]{input,output}})));*/
        //System.out.println("Backpropagation"+Arrays.toString(net.backprop(input,output)[1]));
        //long memoryBefore=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        //List<Matrix[]> training_data=DataLoader.loadTrainingDataAsMatrices(1);
        //List<Matrix[]> test_data=DataLoader.loadTestDataAsMatrices();
        List<INDArray[]> training_data_nd4j=DataLoader.loadTrainingDataAsINDArrays(1);
        List<INDArray[]> test_data_nd4j=DataLoader.loadTestDataAsINDArrays();
        //test_data.stream().limit(100).forEach(m->System.out.println("Elements"+m[1].getElements()));
        /*matrix.getElements().stream().mapToInt(e->e.intValue()).toArray();
        INDArray nd_matrix=Nd4j.create(matrix.getElements().stream().mapToInt(e->e.intValue()).toArray(),
                Arrays.stream(matrix.getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE);
        INDArray nd_matrix1=Nd4j.create(newMatrix.getElements().stream().mapToInt(e->e.intValue()).toArray(),
                Arrays.stream(newMatrix.getShape()).mapToLong(i->(long)i).toArray(),DataType.DOUBLE);
        System.out.println(Arrays.toString(nd_matrix.shape()));
        System.out.println("Result of operation:"+nd_matrix.sub(nd_matrix1).toString());
        System.out.println("nd_matrix:"+nd_matrix.argMax(0).data().asInt()[0]);
        System.out.println(Arrays.toString(Matrices.matrixMultNd4j(matrix, newMatrix).getShape()));
        System.out.println(matrix);
        System.out.println(matrix.argmax());*/
        //System.out.println(test_data.get(0)[1].getElements().get(0).intValue());
        //System.out.println(matrix.argmax()==test_data.get(0)[1].getElements().get(0).intValue());
        /*System.out.println("Training_data memory:"+(memoryBefore-(Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory())));
        System.out.println("Training_data size:"+training_data.size());
        System.out.println("Shape:"+Arrays.toString(training_data.get(0)[0].getShape()));
        System.out.println("Elements:"+training_data.get(0)[0].getElements());
        System.out.println("ElType:"+training_data.get(0)[0].getElType());
        System.out.println("Actual elType:"+training_data.get(0)[0].getElements().get(0).doubleValue()+2);
        System.out.println("Size:"+training_data.get(0)[0].getSize());
        System.out.println(training_data.get(0)[1].getElements());
        System.out.println("Argmax:"+training_data.get(0)[1].argmax());
        System.out.println("Argmax matrix:"+matrix.argmax());
        System.out.println("max matrix:"+matrix.getElements().stream().map(e->e.doubleValue()).max(Double::compare).get());
        System.out.println("Argmax matrix:"+matrix.getElements().stream().map(e->e.doubleValue()).collect(Collectors.toList()).indexOf(matrix.getElements().stream().map(e->e.doubleValue()).max(Double::compare).get()));
        System.out.println("Matrix elType:"+matrix.getElements().get(0).getClass());*/
        //net.backprop(training_data.get(0)[0], training_data.get(0)[1]);
        //net.setWeights(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(784*30).boxed().map(i->i.floatValue()*0.1).collect(Collectors.toList()),new int[]{30,784},Float.class),new Matrix(IntStream.generate(()->1).limit(300).boxed().map(i->i.floatValue()*0.01).collect(Collectors.toList()),new int[]{10,30},Float.class)}));
        //net.setBiases(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(30).boxed().map(i->i.floatValue()*0.01).collect(Collectors.toList()),new int[]{30},Float.class),new Matrix(IntStream.generate(()->1).limit(10).boxed().map(i->i.floatValue()*0.1).collect(Collectors.toList()),new int[]{10},Float.class)}));
        //System.out.println(net.getBiases());
        //System.out.println(net.evaluate(test_data.subList(0, 100)));
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        //net.SGD(training_data, 5, 10, 0.5d, test_data.subList(0, 1000));
        net.SGD(training_data_nd4j, 5, 10, 0.5d, test_data_nd4j.subList(0, 1000));
        //System.out.println(net.backprop(training_data_nd4j.get(0)[0], training_data_nd4j.get(0)[1])[0]);
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        //System.out.println(net.evaluate(test_data.subList(0, 1000)));
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        /*FileReader fr=new FileReader("src\\main\\resources\\data\\mnist_training_data_by_line.json");
        BufferedReader br=new BufferedReader(fr);
        String s=null;
        StringBuilder text=new StringBuilder(br.readLine());
        long beginPoint=System.currentTimeMillis();
        System.out.println(text.substring(text.length()-100));*/
        //System.out.println(net.feedforward(training_data.get(0)[0]));
        /*Matrix training_input=new Matrix(new int[]{10},Double.class);
        Matrix training_output=new Matrix(Arrays.asList(new Double[]{0d,0d,0d,1d,0d}),new int[]{5},Double.class);
        System.out.println("I'm here");
        training_input.setElements(IntStream.generate(()->1).limit(10).boxed().map(i->i.doubleValue()).collect(Collectors.toList())); 
        net.setWeights(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(30).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{3,10},Double.class),new Matrix(IntStream.generate(()->1).limit(15).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{5,3},Double.class)}));
        net.setBiases(Arrays.asList(new Matrix[]{new Matrix(IntStream.generate(()->1).limit(3).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{3},Double.class),new Matrix(IntStream.generate(()->1).limit(5).boxed().map(i->i.doubleValue()).collect(Collectors.toList()),new int[]{5},Double.class)}));
        */
        //System.out.println("Bias grad:"+Arrays.toString(net.backprop(training_input, training_output)[0]));
        //System.out.println("Weight grad:"+Arrays.toString(net.backprop(training_input, training_output)[1]));
        //test_data.stream().limit(100).forEach(m->System.out.println("Elements"+m[1].getElements()));
        /*net.update_mini_batch(Arrays.asList(new Matrix[][]{new Matrix[]{test_data.get(0)[1],training_output}}), 0.5);
        System.out.println("Biases:"+net.getBiases());
        System.out.println("Weights:"+net.getWeights());*/
        //Matrix backprop_input=new Matrix(DoubleStream.generate(()->0.5).limit(10).boxed().collect(Collectors.toList()),new int[]{10},Double.class);
        //System.out.println(Matrices.matrixMultNd4j(net.getWeights().get(0), backprop_input));
        //System.out.println(Arrays.toString(net.backprop(backprop_input, training_output)[0]));
        //List<Matrix[]> training_data=new ArrayList<>();
        //List<Matrix[]> test_data=new ArrayList<>();
        //Random rand=new Random();
        /*for(int i=0;i<10000;i++){
            training_input.setElements(Arrays.asList(new Double[]{rand.nextDouble(),rand.nextDouble(),rand.nextDouble()}));
            training_data.add(new Matrix[]{training_input,training_output});
            if(i<5000){
                training_input.setElements(Arrays.asList(new Double[]{rand.nextDouble(),rand.nextDouble(),rand.nextDouble()}));
                test_data.add(new Matrix[]{training_input,training_output});
            }
        }*/
        //System.out.println(net.feedforward(input));
        //net.SGD(training_data, 10, 50, 1, test_data);
        //System.out.println(net.feedforward(input));
        //int zeros=0;
        /*for(Matrix[] m:training_data){
            Matrix result=net.feedforward(m[0]);
            if(Math.round(result.getElements().get(1).doubleValue())==1 && Math.round(result.getElements().get(0).doubleValue())==0)zeros++;
        }*/
        //System.out.println(zeros);
        //Matrix[][] training_data=DataLoader.loadValidationData();
        //System.out.println(training_data[0][1]);
        //List<Matrix[]> training_data=DataLoader.loadTrainingData();
        //Matrix train_output=training_data.get(0)[1];
        /*System.out.println("Training data size:"+training_data.size());
        System.out.println("Training data datatype:"+(training_data.get(0))[0].getClass());
        System.out.println("Training data example:\n"+(training_data.get(0))[1]);*/
        /*System.out.println("Shape:"+Arrays.toString(train_output.getShape()));
        System.out.println("Size:"+train_output.getSize());
        System.out.println("Axes:"+train_output.getAxes());
        System.out.println(train_output);*/
        /*Matrix w=new Matrix(Arrays.asList(1,2,3,4,5,6,7,8,9),new int[]{3,3},Integer.class);
        Matrix a=new Matrix(Arrays.asList(1,2,3),new int[]{3},Integer.class);
        System.out.println(Matrices.transpose(w));
        System.out.println(Matrices.multiply(a,Matrices.transpose(w)).sum(0));*/
        /*List<Matrix> weights=new ArrayList<>();
        weights.add(new Matrix(IntStream.generate(()->1).limit(9).boxed().collect(Collectors.toList()),new int[]{3,3}, Double.class));
        weights.add(new Matrix(IntStream.generate(()->1).limit(6).boxed().collect(Collectors.toList()),new int[]{2,3}, Double.class));
        List<Matrix> biases=new ArrayList<>();
        biases.add(new Matrix(IntStream.generate(()->1).limit(3).boxed().collect(Collectors.toList()),new int[]{3}, Double.class));
        biases.add(new Matrix(IntStream.generate(()->1).limit(2).boxed().collect(Collectors.toList()),new int[]{2}, Double.class));
        net.setBiases(biases);
        net.setWeights(weights);
        System.out.println("Backpropagation check:"+net.backprop(input, output));*/
    }
    
}
